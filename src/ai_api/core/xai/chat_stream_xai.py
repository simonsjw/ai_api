"""
xAI Streaming Chat Implementation (SDK-based) with Symmetrical Persistence

This module implements real-time token streaming for xAI using the official
SDK async iterator. It is the backend for ``StreamXAIClient.create_chat``
and is called from ``ChatSession`` when ``mode="stream"``.

Design
------
- Yields chunks immediately for low-latency UX.
- Accumulates text; on the final chunk, assembles a complete ``xAIResponse``.
- Persists **only the final response** (request was already persisted by
  the caller / ``persist_chat_turn`` before streaming started).
- Branching metadata (``tree_id``, ``branch_id``, ``parent_response_id``,
  ``sequence``) is carried in the ``request`` object or forwarded via
  ``**kwargs`` when the function signature is extended.

Git-style Branching
-------------------
Because streaming persists only the final turn, branching identifiers must
be present in the incoming ``request`` (populated by ``ChatSession`` or
``xAIClient``). The final ``persist_chat_turn`` call (or equivalent) receives
them so that the ``responses`` table and ``Conversations.active_branch_id``
are updated correctly.

Error Handling
--------------
- SDK / transport errors (gRPC, httpx, authentication, rate limits, timeouts)
  during streaming are wrapped as ``xAIAPIError`` (via ``wrap_xai_api_error``).
- Structured-output parsing failures on the final chunk are wrapped as
  ``xAIClientError`` (logged at WARNING, stream continues).
- Final persistence failures are non-fatal (logged at WARNING with rich
  context via ``wrap_persistence_error``) so the user still receives tokens.
- All raised exceptions inherit from ``AIAPIError`` (directly or indirectly),
  enabling uniform ``except AIAPIError`` handling at higher layers.

See Also
--------
ai_api.core.xai_client.StreamXAIClient
ai_api.core.common.chat_session.ChatSession
ai_api.core.common.persistence.PersistenceManager.persist_chat_turn
ai_api.core.xai.chat_turn_xai
ai_api.core.ollama.chat_stream_ollama
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Type

from pydantic import BaseModel

from ...data_structures.xai_objects import xAIRequest, xAIResponse
from ..common.errors import wrap_client_error, wrap_persistence_error
from ..common.response_struct import create_json_response_spec
from .errors_xai import wrap_xai_api_error

__all__: list[str] = ["generate_stream_and_persist"]


async def generate_stream_and_persist(
    logger: logging.Logger,
    persistence_manager: Any,
    chat: Any,
    request: xAIRequest,
    save_mode: str = "none",
    response_model: Type[BaseModel] | None = None,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    """Stream tokens from xAI, yield them in real time, and persist the
    final assembled response with full branching context.

    Parameters
    ----------
    logger : logging.Logger
        Structured logger (usually from the outer client).
    persistence_manager : PersistenceManager or None
        If provided, the final response is persisted via
        ``persist_chat_turn`` (or the legacy ``persist_response`` path).
    chat : Any
        Async iterator returned by the xAI SDK (``sdk_client.chat.create(...)``).
    request : xAIRequest
        Original request (contains model, messages, branching metadata).
    save_mode : {"none", "json_files", "postgres"}, default "none"
        Persistence backend.
    response_model : type[BaseModel] or None, optional
        Pydantic model for structured output on the final response.
    **kwargs : Any
        Branching identifiers (``tree_id``, ``branch_id``,
        ``parent_response_id``, ``sequence``) forwarded to the final
        persistence call so that ``ChatSession`` and ``edit_history``
        (rebase) work correctly for streaming conversations.

    Yields
    ------
    xAIStreamingChunk or raw SDK chunk
        Tokens as they arrive from the model (for immediate display).

    Raises
    ------
    xAIAPIError
        Any SDK / transport error during streaming (wrapped with full context).
    xAIClientError
        Structured-output parsing failure on the final chunk (non-fatal).

    Notes
    -----
    Only the **final** response is persisted. The request is assumed to have
    been persisted earlier by ``persist_chat_turn`` (the unified entry point).
    Branching context lives in the ``request`` object and is passed through
    to ``persist_chat_turn`` on the final chunk.
    """
    # 1. Create JSON response spec if structured output requested.
    if response_model is not None:
        try:
            spec = create_json_response_spec("xai", response_model)
            request = request.model_copy(
                update={"response_format": spec.to_sdk_response_format()}
            )
        except Exception as exc:
            raise wrap_client_error(
                exc, "Failed to build structured-output specification for xAI stream"
            ) from exc

    full_text: list[str] = []
    final_response: xAIResponse | None = None

    # 2. Stream and accumulate tokens.
    try:
        async for chunk in chat:
            yield chunk
            if chunk.text:
                full_text.append(chunk.text)
            if chunk.is_final:
                final_response = xAIResponse(
                    model=request.model,
                    choices=[
                        {
                            "message": {"content": "".join(full_text)},
                            "finish_reason": chunk.finish_reason,
                        }
                    ],
                    raw=chunk.raw,
                )
                break
    except Exception as exc:
        raise wrap_xai_api_error(
            exc, f"xAI streaming request failed for model '{request.model}'"
        ) from exc

    # 3. Validate structured output on the final response (non-fatal).
    if response_model is not None and final_response is not None:
        try:
            parsed = response_model.model_validate_json("".join(full_text))
            final_response.parsed = parsed
        except Exception as exc:
            wrapped = wrap_client_error(
                exc, "Failed to parse final structured chunk from xAI stream"
            )
            logger.warning(
                "Structured-output parsing failed (continuing)",
                extra={"error": wrapped.message, "details": wrapped.details},
            )

    # 4. Persist final response with branching context (non-fatal on failure).
    if (
        save_mode != "none"
        and persistence_manager is not None
        and final_response is not None
    ):
        try:
            await persistence_manager.persist_chat_turn(
                provider_response=final_response,
                provider_request=request,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    in (
                        "tree_id",
                        "branch_id",
                        "parent_response_id",
                        "sequence",
                        "neutral_history_slice",
                    )
                },
            )
        except Exception as exc:
            wrapped = wrap_persistence_error(
                exc, "Final streaming persistence failed for xAI (continuing)"
            )
            logger.warning(
                wrapped.message,
                extra={"details": wrapped.details, "error": str(exc)},
            )
