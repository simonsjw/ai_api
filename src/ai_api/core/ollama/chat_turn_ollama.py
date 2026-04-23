"""
Fixed version of chat_turn_ollama.py with proper structured output (response_model) support.

Changes:
- Added `response_model` parameter to create_turn_chat_session (matching TurnOllamaClient).
- Converts Pydantic model to OllamaJSONResponseSpec (which sets the top-level "format" key with JSON schema).
- Passes response_format to OllamaRequest.
- Updated docstring.
- Supports both explicit response_model and response_format via **kwargs for flexibility.
"""

from __future__ import annotations

import logging
from typing import Any, Type

import httpx
from pydantic import BaseModel

from ...data_structures.ollama_objects import (
    OllamaJSONResponseSpec,
    OllamaRequest,
    OllamaResponse,
)
from ..common.persistence import PersistenceManager


async def create_turn_chat_session(
    client: Any,  # OllamaClient or similar
    messages: list[dict],
    model: str = "llama3.2",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    save_mode: str = "none",
    response_model: Type[BaseModel] | None = None,
    **kwargs: Any,
) -> OllamaResponse:
    """Create a turn-based chat session with Ollama (updated persistence pattern).

    Supports structured JSON output via:
        - response_model: A Pydantic BaseModel subclass (recommended convenience)
        - response_format: OllamaJSONResponseSpec or raw dict/schema (via **kwargs)

    Example:
        class Person(BaseModel):
            name: str
            age: int

        response = await client.create_chat(
            messages=[{"role": "user", "content": "Extract info"}],
            model="llama3.2",
            response_model=Person,
        )
        # response.parsed will be the validated Person instance (if you post-process)
    """

    logger = client.logger
    logger.info("Creating turn-based Ollama chat", extra={"model": model})

    # Handle structured output
    response_format: OllamaJSONResponseSpec | None = None
    if response_model is not None:
        response_format = OllamaJSONResponseSpec(model=response_model)
    elif "response_format" in kwargs:
        # Allow direct passing of OllamaJSONResponseSpec or dict via **kwargs
        rf = kwargs.pop("response_format")
        if isinstance(rf, dict):
            response_format = OllamaJSONResponseSpec(model=rf)
        elif isinstance(rf, OllamaJSONResponseSpec):
            response_format = rf
        # else: let it be handled by OllamaRequest validation if possible

    # 1. Build request
    request = OllamaRequest(
        model=model,
        input=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        save_mode=save_mode,
        response_format=response_format,
        **kwargs,
    )

    # 2. Persist request (protocol style)
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_request(request)
        except Exception as exc:
            logger.warning(
                "Request persistence failed (continuing)", extra={"error": str(exc)}
            )

    # 3. Call Ollama
    http_client = await client._get_http_client()
    payload = request.to_ollama_dict()
    payload["stream"] = False

    resp = await http_client.post("/api/chat", json=payload)
    resp.raise_for_status()
    raw_data = resp.json()

    response = OllamaResponse.from_dict(raw_data)

    # 4. Persist response (NEW symmetrical protocol style)
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_response(response, request=request)
        except Exception as exc:
            logger.warning(
                "Response persistence failed (continuing)", extra={"error": str(exc)}
            )

    logger.info("Turn-based Ollama chat completed", extra={"model": model})
    return response
