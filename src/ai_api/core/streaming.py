#!/usr/bin/env python3
"""
Streaming utilities for ai_api.

This module provides `StreamingResult` (an accumulator that turns a
stream of `StreamingChunk` objects into a final `LLMResponse`) and the
convenience helper `stream_and_collect`. Together they enable the
recommended pattern for streaming while keeping the low-level
`client.stream()` interface unchanged.

Design priorities (in order):
1. Efficiency — single-pass accumulation, no unnecessary copies.
2. Clarity — explicit separation of chunk collection from persistence.
3. Readability — every function ≤ 40 lines, full NumPy-style docstrings,
   inline comments after column 90.

The module deliberately does **not** perform persistence itself; it
returns a ready `LLMResponse` so the caller (or concrete client) can
decide when to save. This keeps streaming lightweight and gives full
control over timing of database writes.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..clients.base import BaseAsyncProviderClient

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .request import LLMRequest
from .response import LLMResponse, ProviderLiteral, StreamingChunk

if TYPE_CHECKING:
    from ..clients.base import BaseAsyncProviderClient


@dataclass
class StreamingResult:
    """
    Accumulator for streaming chunks that produces a final LLMResponse.

    Use this when you want to both consume the stream in real time and
    obtain a complete response object at the end (for persistence,
    logging, or return to caller).
    """

    request: LLMRequest
    provider: ProviderLiteral
    chunks: list[StreamingChunk] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)                              # for models with <think>

    def add_chunk(self, chunk: StreamingChunk) -> None:
        """
        Append a chunk and update internal state.

        Parameters
        ----------
        chunk : StreamingChunk
            Chunk received from client.stream().
        """
        self.chunks.append(chunk)
        if chunk.delta_text:
            self.text_parts.append(chunk.delta_text)
        if chunk.tool_call_delta:
            self.tool_calls.append(chunk.tool_call_delta)
            # reasoning support (added for future models)
        if hasattr(chunk, "reasoning_delta") and chunk.reasoning_delta:
            self.reasoning_parts.append(chunk.reasoning_delta)

    def to_response(
        self, continuation_token: Any = None, raw_final: dict | None = None
    ) -> LLMResponse:
        """
        Convert accumulated state into a final immutable LLMResponse.

        Returns
        -------
        LLMResponse
            Ready for persistence or return to caller.
        """
        full_text = "".join(self.text_parts).strip()
        reasoning_text = "".join(self.reasoning_parts).strip()

        extra = {"reasoning": reasoning_text} if reasoning_text else {}

        return LLMResponse(
            id=raw_final.get("id", "stream-unknown") if raw_final else "stream-unknown",
            created_at=raw_final.get("created_at", 0) if raw_final else 0,
            model=self.request.model,
            provider=self.provider,
            text=full_text,
            tool_calls=self.tool_calls,
            usage=raw_final.get("usage") if raw_final else None,
            continuation_token=continuation_token,
            raw=raw_final or {},
            extra=extra,
        )


async def stream_and_collect(
    client: "BaseAsyncProviderClient",
    request: LLMRequest,
    save_on_complete: bool = False,
) -> LLMResponse:
    """
    Stream a request, collect chunks, and optionally persist the final
    response.

    This is the recommended high-level streaming helper.

    Parameters
    ----------
    client : BaseAsyncProviderClient
        Either Grok or Ollama client.
    request : LLMRequest
        The unified request.
    save_on_complete : bool, optional
        If True, the final response is saved to JSON or PostgreSQL
        exactly once when the stream finishes.

    Returns
    -------
    LLMResponse
        Complete response (with reasoning captured if present).
    """
    result = StreamingResult(request=request, provider=client.provider_name)
    raw_final: dict | None = None
    continuation_token: Any = None

    # Fixed: Pyrefly now recognises this as a proper async iterable
    async for chunk in client.stream(request):
        result.add_chunk(chunk)
        if chunk.finished:
            raw_final = chunk.raw_chunk
            continuation_token = getattr(chunk, "continuation_token", None)

    final_response = result.to_response(continuation_token, raw_final)

    # Fixed: pass BOTH objects for perfect 1:1 request ↔ response linking
    if save_on_complete and hasattr(client, "_persist_streamed_response"):
        await client._persist_streamed_response(request, final_response)

    return final_response
