"""Streaming handler for xAI API responses.

This module defines the ``xAIStreamHandler`` class responsible for
managing asynchronous streaming generations from the xAI xAI API.
It yields individual chunks to the caller in real time while internally
accumulating the full response. A single final response is persisted to
the database upon completion of the stream.

This design prevents per-chunk database writes and keeps streaming
orchestration logic cleanly separated from core client and persistence
concerns.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

# Local project imports
from ...data_structures.xai_objects import (
    LLMStreamingChunkProtocol,
    xAIRequest,
    xAIStreamingChunk,
)
from .xai_errors import wrap_grok_api_error

__all__: list[str] = [
    "xai_stream",
    "generate_stream_and_persist",
]


async def xai_stream(
    self, chat: Any, request: xAIRequest
) -> AsyncIterator[LLMStreamingChunkProtocol]:
    """Streaming helper (wrapped for logging)."""
    try:
        async for _full, chunk in chat.stream():
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
        raise wrap_grok_api_error(exc, "Streaming generation failed") from exc


async def generate_stream_and_persist(
    self,
    chat: Any,
    request: xAIRequest,
    persist_info: dict,
) -> AsyncIterator[LLMStreamingChunkProtocol]:
    """Yields streaming chunks to the caller AND persists ONE final response row at completion.
    Accumulation avoids per-chunk row explosion while still capturing the complete output."""
    full_text: list[str] = []
    final_finish_reason: str | None = None

    async for chunk in self._grok_stream(chat, request):
        yield chunk
        if chunk.text:
            full_text.append(chunk.text)
        if getattr(chunk, "is_final", False):
            final_finish_reason = getattr(chunk, "finish_reason", None)

            # Build final result compatible with _persist_response
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

    await self._persist_response(
        persist_info["request_id"],
        persist_info["request_tstamp"],
        final_result,
        request,
    )
