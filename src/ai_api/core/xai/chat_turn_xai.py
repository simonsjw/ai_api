"""Turn-based (non-streaming) chat functionality for the xAI API."""

import uuid
from datetime import datetime
from typing import Any, Optional

from xai_sdk import AsyncClient
from xai_sdk.chat import system, user                                                     # SDK message builders

from ...data_structures.xai_objects import (                                              # exact path as provided
    SaveMode,
    xAIInput,
    xAIJSONResponseSpec,
    xAIRequest,
    xAIResponse,
)
from ..xai_client import BaseXAIClient                                                    # type reference only
from .common_xai import _generate_non_streaming
from .errors_xai import wrap_infopypg_error, xAIClientError
from .persistence_xai import xAIPersistenceManager

__all__: list[str] = ["create_turn_chat_session"]


async def create_turn_chat_session(
    client: BaseXAIClient,
    sdk_client: AsyncClient,
    messages: list[dict[str, Any]],
    model: str = "grok-4",
    *,
    response_model: xAIJSONResponseSpec | None = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    save_mode: SaveMode = "none",
    **kwargs: Any,
) -> xAIResponse:
    """Create a stateful turn-based chat completion using the official xAI SDK.

    This replaces the previous custom HTTP implementation. The SDK manages
    conversation state internally via the returned chat object, which is attached
    to the response for easy continuation in subsequent turns.

    Args:
        client: BaseXAIClient instance (provides logger and persistence manager).
        sdk_client: The official AsyncClient from xai_sdk.
        messages: Full conversation history (list of dicts with 'role' and 'content').
        model: Model identifier (default: "grok-3").
        temperature, max_tokens, top_p: Standard generation parameters.
        save_mode: Persistence behaviour ("none" or "postgres").
        **kwargs: Additional parameters passed through to the SDK chat.create().

    Returns:
        xAIResponse instance (identical contract to the previous implementation).
        The response object also contains a private `_sdk_chat` attribute for
        stateful continuation if the caller wishes to avoid rebuilding history.
    """
    client.logger.info(
        "Creating turn-based xAI chat (official SDK)",
        extra={"model": model, "message_count": len(messages)},
    )

    # 1. Build xAIRequest (reuses all existing validation + response_spec logic)
    xai_request = xAIRequest(
        model=model,
        input=xAIInput.from_list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        response_format=response_model,                                                   # your existing spec handling
        **{k: v for k, v in kwargs.items() if k not in {"save_mode", "response_model"}},
    )

    # 2. Persist request BEFORE network call (unchanged contract)
    request_id: uuid.UUID | None = None
    request_tstamp: datetime | None = None
    if save_mode == "postgres" and client.persistence_manager is not None:
        try:
            (
                request_id,
                request_tstamp,
            ) = await client.persistence_manager.persist_request(xai_request)
        except Exception as exc:                                                          # noqa: BLE001
            client.logger.error("Failed to persist request (non-fatal)", exc_info=True)

            # 3. Create SDK chat using the new helper
    chat_kwargs = xai_request.to_sdk_chat_kwargs()
    chat = sdk_client.chat.create(**chat_kwargs)

    # 4. Generate response
    try:
        sdk_response = await chat.sample()
    except Exception as exc:
        raise wrap_infopypg_error(
            exc, "Failed to generate turn-based response via xAI SDK"
        ) from exc

    # 5. Convert and attach structured / stateful data

    if response_model is not None:
        # First convert the raw SDK response (required for .parsed and ._sdk_chat)
        response = xAIResponse.from_sdk(sdk_response)

        # Now safely parse using the full xAIResponse object
        parsed: xAIJSONResponseSpec = response_model.from_xai_response(response)
        response.parsed = parsed
    else:
        # No structured model requested
        response = xAIResponse.from_sdk(sdk_response)
        parsed = None

        # Attach the stateful SDK chat object for continuation
    response._sdk_chat = chat

    # 6. Persist response
    if (
        save_mode == "postgres"
        and client.persistence_manager is not None
        and request_id is not None
        and request_tstamp is not None
    ):
        try:
            await client.persistence_manager.persist_response(
                request_id=request_id,
                request_tstamp=request_tstamp,
                api_result={"raw": sdk_response},
                request=xai_request,
            )
        except Exception as exc:                                                          # noqa: BLE001
            client.logger.error("Failed to persist response (non-fatal)", exc_info=True)

    client.logger.info(
        "Turn-based chat completed successfully (official SDK)",
        extra={"response_id": getattr(sdk_response, "id", None)},
    )
    return response
