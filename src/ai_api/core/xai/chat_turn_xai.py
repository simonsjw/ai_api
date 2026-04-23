"""
Example of updated chat_turn_xai.py using the new symmetrical persistence pattern.

Key change:
    OLD: await persistence.persist_response(request_id=..., request_tstamp=..., api_result=..., request=request)
    NEW: await persistence.persist_response(response, request=request)
"""

from __future__ import annotations

import logging
from typing import Any, Type

from pydantic import BaseModel

from ...data_structures.xai_objects import xAIRequest, xAIResponse
from ..common.persistence import PersistenceManager
from ..common.response_struct import create_json_response_spec


async def create_turn_chat_session(
    client: Any,
    sdk_client: Any,
    messages: list[dict],
    model: str = "grok-4",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    save_mode: str = "none",
    response_model: Type[BaseModel] | None = None,
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

    # 2. Include specification for response if provided.
    if response_model is not None:
        spec = create_json_response_spec("xai", response_model)
        # Attach to the request (adjust field name to match your xAIRequest)
        request = request.model_copy(
            update={"response_format": spec.to_sdk_response_format()}
        )

        # 3. Persist request (protocol style)
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

            # 4. Call the actual LLM
            # (simplified - in real code this would use the xAI SDK)
    chat = sdk_client.chat.create(
        model=request.model,
        **request.to_sdk_chat_kwargs(),
    )
    raw_response = await chat.completions.create()
    response = xAIResponse.from_sdk(raw_response)

    # 5. Validate the response if a response specification is provided.
    if response_model is not None:
        try:
            parsed = response_model.model_validate_json(response.text)
            response.parsed = parsed
        except Exception as exc:
            client.logger.warning(
                "Failed to parse structured response", extra={"error": str(exc)}
            )

            # 6. Persist response (NEW symmetrical protocol style)
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_response(response, request=request)
        except Exception as exc:
            logger.warning(
                "Response persistence failed (continuing)", extra={"error": str(exc)}
            )

    logger.info("Turn-based xAI chat completed", extra={"model": model})
    return response
