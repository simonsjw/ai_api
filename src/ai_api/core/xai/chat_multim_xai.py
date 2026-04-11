"""Multimodal chat functionality for the xAI API (Responses API).

This module provides the implementation for ``MultimXAIClient.create_chat(...)``.
It is functionally identical to ``create_turn_chat`` but is maintained as a
separate file for clarity and future extensibility (e.g., additional multimodal
validation, media preprocessing, or model-specific constraints).

Multimodal support is provided natively by the underlying data structures:
- ``xAIMessage.content`` can be ``str`` or ``list[dict]`` containing
  ``{"type": "input_image", "image_url": "..."}`` or ``{"type": "input_file", ...}``.
- ``xAIRequest.has_media()`` and persistence media handling are automatically
  invoked when applicable.

All other behaviours (Responses API endpoint, persistence hooks, structured
logging, custom error wrapping) are kept identical to the turn-based path for
consistency and maintainability.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from ...data_structures.xai_objects import (                                              # exact path as provided
    SaveMode,
    xAIInput,
    xAIRequest,
    xAIResponse,
)
from ..xai_client import BaseXAIClient
from .common_xai import _generate_non_streaming
from .errors_xai import wrap_infopypg_error, xAIClientError
from .persistence_xai import xAIPersistenceManager

__all__: list[str] = ["create_multim_chat"]


async def create_multim_chat(
    client: BaseXAIClient,
    messages: list[dict[str, Any]],
    model: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    save_mode: SaveMode = "none",
    **kwargs: Any,
) -> xAIResponse:
    """Create a multimodal chat completion using the modern Responses API.

    Supports text-only or multimodal messages (images, files) via the
    same canonical request pipeline as turn-based chat. Persistence,
    structured logging, and error handling are fully integrated.
    """
    # 1. Build the canonical request object
    xai_input = xAIInput.from_list(messages)
    request = xAIRequest(
        input=xai_input,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        save_mode=save_mode,
        **kwargs,
    )

    # 2. Prepare the exact payload for the Responses API
    request_kwargs = request.to_api_kwargs()

    request_id: uuid.UUID | None = None
    request_tstamp: datetime | None = None

    # Structured log at INFO level (consistent with turn-based chat)
    client.logger.info(
        "Creating multimodal xAI chat",
        extra={
            "obj": {
                "model": model,
                "save_mode": save_mode,
                "has_media": request.has_media(),
            }
        },
    )

    # === PERSISTENCE: REQUEST (before API call) ===
    if save_mode == "postgres" and client.persistence_manager is not None:
        try:
            (
                request_id,
                request_tstamp,
            ) = await client.persistence_manager.persist_request(request)
        except Exception as exc:                                                          # persistence failure must not block the API call
            client.logger.warning(
                "Request persistence failed (continuing with API call)",
                extra={"obj": {"model": model, "error": str(exc)}},
            )
            # Still surface the error via the proper hierarchy so callers can handle it
            raise wrap_infopypg_error(
                exc, "Failed to persist multimodal request"
            ) from exc
        # (request_id / request_tstamp remain None → response persistence is skipped)

    try:
        raw_response = await _generate_non_streaming(
            client=client,
            endpoint="/v1/responses",
            json_data=request_kwargs,
        )

        response = xAIResponse.from_dict(raw_response)

        # === PERSISTENCE: RESPONSE (after successful call) ===
        if (
            save_mode == "postgres"
            and client.persistence_manager is not None
            and request_id is not None
            and request_tstamp is not None
        ):
            await client.persistence_manager.persist_response(
                request_id=request_id,
                request_tstamp=request_tstamp,
                api_result=raw_response,
                request=request,                                                          # passes has_media() information
            )

        client.logger.info(
            "Multimodal chat completed successfully",
            extra={"obj": {"model": model, "has_media": request.has_media()}},
        )

        return response

    except Exception as exc:
        client.logger.error(
            "Multimodal chat creation failed",
            extra={"obj": {"model": model, "error": str(exc)}},
        )
        raise xAIClientError(f"Multimodal chat creation failed: {exc}") from exc
