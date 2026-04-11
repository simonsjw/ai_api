"""Turn-based (non-streaming) chat functionality for the xAI API."""

import uuid
from datetime import datetime
from typing import Any, Optional

from ...data_structures.xai_objects import (                                              # exact path as provided
    SaveMode,
    xAIInput,
    xAIRequest,
    xAIResponse,
)
from ..xai_client import BaseXAIClient                                                    # type reference only
from .common_xai import _generate_non_streaming
from .errors_xai import wrap_infopypg_error, xAIClientError
from .persistence_xai import xAIPersistenceManager


async def create_turn_chat(
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
    """Create a standard turn-based (non-streaming) chat completion.

    All logic is now expressed in terms of the official data structures
    defined in data_structures/xai_objects.py.
    """
    # 1. Build the canonical request object (mirrors your existing design)
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

    # 2. Prepare the exact kwargs expected by the xAI API
    request_kwargs = request.to_api_kwargs()

    request_id: uuid.UUID | None = None
    request_tstamp: datetime | None = None

    client.logger.info(
        "Creating turn-based xAI chat",
        extra={"obj": {"model": model, "save_mode": save_mode}},
    )

    # === PERSISTENCE: REQUEST (before API call) ===
    if save_mode == "postgres" and client.persistence_manager is not None:
        try:
            (
                request_id,
                request_tstamp,
            ) = await client.persistence_manager.persist_request(request)
        except Exception as exc:                                                          # persistence failure should not block the API call
            client.logger.warning(
                "Request persistence failed (continuing)",
                extra={"obj": {"model": model, "error": str(exc)}},
            )
            # Still raise a wrapped error so the caller knows persistence failed
            if isinstance(exc, Exception):
                raise wrap_infopypg_error(exc, "Failed to persist request") from exc
            request_id = request_tstamp = None

    try:
        raw_response = await _generate_non_streaming(
            client=client,
            json_data=request_kwargs,
            endpoint="/v1/responses",
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
                request=request,
            )

        client.logger.info(
            "Turn-based chat completed successfully", extra={"obj": {"model": model}}
        )
        return response

    except Exception as exc:
        client.logger.error(
            "Turn-based chat creation failed", extra={"obj": {"model": model}}
        )
        raise xAIClientError(f"Turn-based chat creation failed: {exc}") from exc
