"""
Example of updated chat_turn_ollama.py using the new symmetrical persistence pattern.

Key change:
    OLD: await persistence.persist_response(request_id=..., request_tstamp=..., api_result=..., request=request)
    NEW: await persistence.persist_response(response, request=request)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ...data_structures.ollama_objects import OllamaRequest, OllamaResponse
from ..common.persistence import PersistenceManager


async def create_turn_chat_session(
    client: Any,                                                                          # OllamaClient or similar
    messages: list[dict],
    model: str = "llama3.2",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    save_mode: str = "none",
    **kwargs: Any,
) -> OllamaResponse:
    """Create a turn-based chat session with Ollama (updated persistence pattern)."""

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

    # 2. Persist request (protocol style)
    request_id = request_tstamp = None
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            (
                request_id,
                request_tstamp,
            ) = await client.persistence_manager.persist_request(request)
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
