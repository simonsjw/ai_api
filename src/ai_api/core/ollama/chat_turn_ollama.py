"""
Example of updated chat_turn_ollama.py using the new symmetrical persistence pattern.

Key change:
    OLD: await persistence.persist_response(request_id=..., request_tstamp=..., api_result=..., request=request)
    NEW: await persistence.persist_response(response, request=request)
"""

from __future__ import annotations

import logging
from typing import Any, Type

import httpx
from pydantic import BaseModel

from ...data_structures.ollama_objects import OllamaRequest, OllamaResponse
from ..common.persistence import PersistenceManager
from ..common.response_struct import create_json_response_spec


async def create_turn_chat_session(
    client: Any,
    messages: list[dict],
    model: str = "llama3.2",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    save_mode: str = "none",
    response_model: Type[BaseModel] | None = None,
    **kwargs: Any,
) -> OllamaResponse:
    """Create a turn-based chat session with Ollama (with structured output support)."""

    logger = client.logger
    logger.info("Creating turn-based Ollama chat", extra={"model": model})

    # 1. Build request
    request = OllamaRequest(
        model=model,
        input=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        save_mode=save_mode,
        **kwargs,
    )

    # 2. Handle structured output (correct way)
    if response_model is not None:
        spec = create_json_response_spec("ollama", response_model)
        # Update the request object so persistence sees the format
        request = request.model_copy(
            update={
                "response_format": spec
            }                                                                             # or add a dedicated `format` field if you prefer
        )

        # 3. Persist request (using protocol object - this is correct)
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_request(request)
        except Exception as exc:
            logger.warning(
                "Request persistence failed (continuing)", extra={"error": str(exc)}
            )

            # 4. Build final payload for Ollama API
    payload = request.to_ollama_dict()
    payload["stream"] = False

    # Add format if structured output was requested
    if response_model is not None:
        spec = create_json_response_spec("ollama", response_model)
        payload["format"] = spec.to_ollama_format()

        # 5. Call Ollama
    http_client = await client._get_http_client()
    resp = await http_client.post("/api/chat", json=payload)
    resp.raise_for_status()
    raw_data = resp.json()

    response = OllamaResponse.from_dict(raw_data)

    # 6. Persist response (using protocol object - this is correct)
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_response(response, request=request)
        except Exception as exc:
            logger.warning(
                "Response persistence failed (continuing)", extra={"error": str(exc)}
            )

    logger.info("Turn-based Ollama chat completed", extra={"model": model})
    return response
