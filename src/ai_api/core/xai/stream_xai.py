"""Streaming handler for xAI API responses – official SDK path."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ...data_structures.xai_objects import (
    LLMStreamingChunkProtocol,
    SaveMode,
    xAIRequest,
    xAIStreamingChunk,
)
from .errors_xai import wrap_xai_api_error

__all__: list[str] = ["xai_stream", "generate_stream_and_persist"]


async def xai_stream(
    self, chat: Any, request: xAIRequest
) -> AsyncIterator[LLMStreamingChunkProtocol]:
    """Low-level SDK streaming wrapper."""
    try:
        async for _full, chunk in chat.stream():                                          # official SDK pattern
            yield xAIStreamingChunk(
                text=getattr(chunk, "content", ""),
                finish_reason=getattr(chunk, "finish_reason", None),
                is_final=getattr(chunk, "is_final", False),
                raw={"chunk": chunk},
            )
    except Exception as exc:
        self.logger.error(
            "Streaming generation failed", extra={"obj": {"error": str(exc)}}
        )
        raise wrap_xai_api_error(exc, "Streaming generation failed") from exc


async def generate_stream_and_persist(
    self,
    chat: Any,
    request: xAIRequest,
    save_mode: SaveMode = "none",
) -> AsyncIterator[LLMStreamingChunkProtocol]:
    """Yields chunks in real time + persists ONE final response row at completion.

    Uses the injected persistence_manager (keeps your existing mechanism valid).
    """
    if save_mode == "postgres" and self.persistence_manager is not None:
        try:
            request_id, request_tstamp = await self.persistence_manager.persist_request(
                request
            )
        except Exception as exc:
            self.logger.warning(
                "Request persistence failed (continuing)",
                extra={"obj": {"error": str(exc)}},
            )
            request_id = request_tstamp = None
    else:
        request_id = request_tstamp = None

    full_text: list[str] = []
    final_finish_reason: str | None = None

    async for chunk in xai_stream(self, chat, request):
        yield chunk
        if chunk.text:
            full_text.append(chunk.text)
        if getattr(chunk, "is_final", False):
            final_finish_reason = getattr(chunk, "finish_reason", None)

            # Single final persistence (exactly as designed)
    if (
        save_mode == "postgres"
        and self.persistence_manager is not None
        and request_id is not None
        and request_tstamp is not None
    ):
        final_result = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "".join(full_text)}],
                }
            ],
            "model": request.model,
            "finish_reason": final_finish_reason,
            "raw": {"accumulated_text": "".join(full_text)},
        }
        await self.persistence_manager.persist_response(
            request_id=request_id,
            request_tstamp=request_tstamp,
            api_result=final_result,
            request=request,
        )
