"""
Ollama streaming chat with final-response persistence (Git-style branching).

This module provides the coroutine ``generate_stream_and_persist`` used by
``StreamOllamaClient.create_chat``. It yields ``OllamaStreamingChunk``
objects in real time while accumulating the full assistant message. When
the stream ends, the accumulated data is converted to an ``OllamaResponse``
and persisted **exactly once** via ``PersistenceManager.persist_chat_turn``.

Branching support
-----------------
- Branching metadata (``tree_id``, ``branch_id``, ``parent_response_id``,
  ``sequence``) is accepted via ``**kwargs`` and forwarded verbatim to
  ``persist_chat_turn``.
- Only the **final** response is ever written (intermediate chunks are
  discarded to avoid thousands of rows for long generations).
- The design guarantees exactly-once persistence even if the stream is
  cancelled or errors (best-effort partial persistence on failure).

See Also
--------
chat_turn_ollama : non-streaming counterpart (identical persistence path).
ai_api.core.common.chat_session.ChatSession : recommended high-level API
    that most callers should use instead of invoking this coroutine directly.
ai_api.core.common.persistence.PersistenceManager.persist_chat_turn
    The unified method that receives the final response + branching metadata.
ai_api.core.common.persistence.PersistenceManager.reconstruct_neutral_branch
    Rebuilds full conversation history from the ``parent_response_id`` chain.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...data_structures.ollama_objects import (
    OllamaRequest,
    OllamaResponse,
    OllamaStreamingChunk,
)
from ..common.persistence import PersistenceManager


async def generate_stream_and_persist(
    logger: logging.Logger,
    persistence_manager: PersistenceManager | None,
    http_client: httpx.AsyncClient,
    request: OllamaRequest,
    save_mode: str = "none",
    **kwargs: Any,
) -> AsyncIterator[OllamaStreamingChunk]:
    """Stream tokens from Ollama and persist the final response.

    This coroutine is the streaming counterpart of ``create_turn_chat_session``.
    It yields ``OllamaStreamingChunk`` objects in real time while
    accumulating the full assistant message.  When the stream ends, the
    accumulated data is turned into an ``OllamaResponse`` and persisted via
    the single ``persist_chat_turn`` entry point.

    Parameters
    ----------
    logger : logging.Logger
        Structured logger (passed from the client).
    persistence_manager : PersistenceManager or None
        If ``None`` or ``save_mode="none"`` no persistence occurs.
    http_client : httpx.AsyncClient
        Pre-configured client pointing at the Ollama host.
    request : OllamaRequest
        The fully constructed request (already contains ``save_mode``,
        temperature, structured-output spec, etc.).
    save_mode : {"none", "json_files", "postgres"}, default "none"
        Passed through to ``persist_chat_turn``.
    **kwargs
        Additional branching parameters (``tree_id``, ``branch_id``,
        ``parent_response_id``, ``sequence``) forwarded to
        ``persist_chat_turn``.

    Yields
    ------
    OllamaStreamingChunk
        One chunk per token (or tool-call delta).  The final chunk has
        ``is_final=True`` and contains the complete ``finish_reason`` and
        usage statistics.

    Raises
    ------
    httpx.HTTPStatusError
        Propagated from the streaming POST if Ollama returns an error.
    Exception
        Any persistence error is logged at WARNING but does not abort the
        stream.

    Examples
    --------
    Typical usage inside StreamOllamaClient
    >>> async for chunk in generate_stream_and_persist(
    ...     logger,
    ...     pm,
    ...     http_client,
    ...     request,
    ...     save_mode="postgres",
    ...     tree_id=ctx["tree_id"],
    ...     branch_id=ctx["branch_id"],
    ... ):
    ...     print(chunk.text, end="", flush=True)
    """
    payload = request.to_ollama_dict()
    payload["stream"] = True

    accumulated: dict[str, Any] = {
        "model": request.model,
        "message": {"role": "assistant", "content": ""},
        "done": False,
    }
    usage: dict[str, Any] = {}

    try:
        async with http_client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk_data = json.loads(line)

                # Merge incremental content
                if "message" in chunk_data and "content" in chunk_data["message"]:
                    accumulated["message"]["content"] += chunk_data["message"][
                        "content"
                    ]

                # Capture final metadata on the last chunk
                if chunk_data.get("done"):
                    accumulated.update(chunk_data)
                    usage = {
                        "total_duration": chunk_data.get("total_duration"),
                        "load_duration": chunk_data.get("load_duration"),
                        "prompt_eval_count": chunk_data.get("prompt_eval_count"),
                        "eval_count": chunk_data.get("eval_count"),
                    }
                    break

                # Yield incremental chunk
                yield OllamaStreamingChunk.from_dict(chunk_data)

        # ------------------------------------------------------------------
        # Build final response and persist via unified method
        # ------------------------------------------------------------------
        final_response = OllamaResponse.from_dict(accumulated)
        final_response.usage = usage                                                      # attach telemetry

        if save_mode != "none" and persistence_manager is not None:
            try:
                await persistence_manager.persist_chat_turn(
                    provider_response=final_response,
                    provider_request=request,
                    kind="chat",
                    branching=True,
                    **kwargs,                                                             # tree/branch/parent/sequence
                )
            except Exception as exc:
                logger.warning(
                    "Streaming final-response persistence failed",
                    extra={"error": str(exc)},
                )

        # Emit one final synthetic chunk so callers know the stream ended
        yield OllamaStreamingChunk(
            text="",
            finish_reason=accumulated.get("done_reason", "stop"),
            tool_calls_delta=None,
            is_final=True,
            raw=accumulated,
        )

    except Exception:
        # Best-effort persistence of whatever we have so far
        if (
            save_mode != "none"
            and persistence_manager is not None
            and accumulated.get("message", {}).get("content")
        ):
            try:
                partial = OllamaResponse.from_dict(accumulated)
                await persistence_manager.persist_chat_turn(
                    provider_response=partial,
                    provider_request=request,
                    kind="chat",
                    branching=True,
                    **kwargs,
                )
            except Exception:
                pass
        raise
