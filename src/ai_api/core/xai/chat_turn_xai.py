"""Turn-based (non-streaming) chat functionality for the xAI API."""

import uuid
from datetime import datetime
from typing import Any, Optional

from xai_sdk import AsyncClient
from xai_sdk.chat import system, user

from ...data_structures.xai_objects import (
    SaveMode,
    xAIInput,
    xAIJSONResponseSpec,
    xAIRequest,
    xAIResponse,
)
from ..common.persistence import PersistenceManager
from ..xai_client import BaseXAIClient
from .errors_xai import wrap_infopypg_error, xAIClientError

__all__: list[str] = ["create_turn_chat_session"]


async def create_turn_chat_session(
    client: "BaseXAIClient",                                                              # forward reference – defined in xai_client.py
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
    """Create a stateful turn-based chat using the official xAI SDK.

    Fully supports multimodal media (images/files) – see xAIRequest docstring for attachment format.
    Media files are automatically saved to disk when persistence_manager.media_root is configured.
    """
    client.logger.info(
        "Creating turn-based xAI chat (official SDK)",
        extra={
            "model": model,
            "message_count": len(messages),
            "has_media": any(
                msg.get("content")
                and isinstance(msg.get("content"), list)
                and any(
                    isinstance(part, dict)
                    and part.get("type") in ("input_image", "input_file")
                    for part in (
                        msg.get("content")
                        if isinstance(msg.get("content"), list)
                        else []
                    )
                )
                for msg in messages
            ),
        },
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
        response_raw: xAIResponse = xAIResponse.from_sdk(sdk_response)                    # temporary
        parsed: xAIJSONResponseSpec = response_model.from_xai_response(response_raw)
        response = xAIResponse.from_sdk(                                                  # final immutable instance
            sdk_response=sdk_response, parsed=parsed, sdk_chat=chat
        )
    else:
        response = xAIResponse.from_sdk(
            sdk_response=sdk_response, parsed=None, sdk_chat=chat
        )

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
