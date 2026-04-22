"""
Example of updated chat_turn_xai.py using the new symmetrical persistence pattern.

Key change:
    OLD: await persistence.persist_response(request_id=..., request_tstamp=..., api_result=..., request=request)
    NEW: await persistence.persist_response(response, request=request)
"""

from __future__ import annotations

import logging
from typing import Any

from ...data_structures.xai_objects import xAIRequest, xAIResponse
from ..common.persistence import PersistenceManager


async def create_turn_chat_session(
    client: Any,                                                                          # XAIClient or similar
    sdk_client: Any,
    messages: list[dict],
    model: str = "grok-4",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    save_mode: str = "none",
    **kwargs: Any,
) -> xAIResponse:
    """Create a turn-based chat session with xAI (updated persistence pattern)."""

    logger = client.logger
    logger.info("Creating turn-based xAI chat", extra={"model": model})

    # 1. Build request
    request = xAIRequest(
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

            # 3. Call the actual LLM
            # (simplified - in real code this would use the xAI SDK)
    raw_response = await sdk_client.chat.create(...)                                      # placeholder
    response = xAIResponse.from_sdk(raw_response)

    # 4. Persist response (NEW symmetrical protocol style)
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_response(response, request=request)
        except Exception as exc:
            logger.warning(
                "Response persistence failed (continuing)", extra={"error": str(exc)}
            )

    logger.info("Turn-based xAI chat completed", extra={"model": model})
    return response
