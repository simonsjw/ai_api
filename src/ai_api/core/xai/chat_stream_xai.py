"""
xAI streaming chat implementation (SDK-based) with symmetrical persistence.

This module implements token-by-token streaming for xAI using the official
SDK's async iterator. It is the concrete implementation behind the streaming
path of ``xAIClient``.

High-level view of xAI streaming
--------------------------------
- Consumes an async iterator from the xAI SDK (``async for chunk in chat``).
- Yields raw chunks immediately for real-time UX.
- Accumulates text; on the final chunk, builds a complete ``xAIResponse``
  and (optionally) validates it against a ``response_model``.
- Persists only the final assembled response (the request was persisted
  by the caller before streaming started).

Comparison with Ollama streaming
--------------------------------
- xAI: SDK async iterator, final response constructed from accumulated
  chunks + raw metadata, supports thinking mode and richer error types.
- Ollama: native HTTP streaming with per-chunk telemetry (durations),
  local execution, more generation parameters.
- Both use the identical persistence pattern (persist final response only)
  and the same structured-output helper from ``common/response_struct.py``.

See Also
--------
ai_api.core.xai_client
ai_api.core.xai.chat_turn_xai
    The non-streaming counterpart.
ai_api.core.ollama.chat_stream_ollama
    The Ollama streaming implementation (native HTTP).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Type

from pydantic import BaseModel

from ...data_structures.xai_objects import xAIRequest, xAIResponse, xAIStreamingChunk
from ..common.persistence import PersistenceManager
from ..common.response_struct import create_json_response_spec


async def generate_stream_and_persist(
    logger: logging.Logger,
    persistence_manager: Any,
    chat: Any,
    request: xAIRequest,
    save_mode: str = "none",
    response_model: Type[BaseModel] | None = None,
) -> AsyncIterator[Any]:
    """Stream tokens from xAI and persist the final assembled response.

    Parameters
    ----------
    logger : logging.Logger
    persistence_manager : PersistenceManager or None
    chat : Any
        The xAI SDK async chat iterator (already started).
    request : xAIRequest
        The original request (used for context and persistence linking).
    save_mode : {"none", "json_files", "postgres"}, optional
    response_model : type[pydantic.BaseModel] or None, optional
        If supplied, the accumulated text is validated and attached as
        ``.parsed`` on the final response.

    Yields
    ------
    xAIStreamingChunk (or raw SDK chunk)
        Real-time tokens.
    """

    # 1. Create JSON response spec.
    if response_model is not None:
        spec = create_json_response_spec("xai", response_model)
        request = request.model_copy(
            update={"response_format": spec.to_sdk_response_format()}
        )

    full_text: list[str] = []
    final_response: xAIResponse | None = None

    # 2. Collect streaming chunks.
    async for chunk in chat:
        yield chunk
        if chunk.text:
            full_text.append(chunk.text)
        if chunk.is_final:
            # Build final response object from accumulated data
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

            # 3. validate with response specification if provided.
    if response_model is not None and final_response is not None:
        try:
            parsed = response_model.model_validate_json("".join(full_text))
            final_response.parsed = parsed
        except Exception as exc:
            logger.warning(
                "Failed to parse final structured chunk", extra={"error": str(exc)}
            )

            # 4. Persist the final response (symmetrical protocol style)
    if (
        save_mode != "none"
        and persistence_manager is not None
        and final_response is not None
    ):
        try:
            await persistence_manager.persist_response(final_response, request=request)
        except Exception as exc:
            logger.warning(
                "Response persistence failed (continuing)", extra={"error": str(exc)}
            )
