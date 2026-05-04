"""
Ollama streaming chat with final-response persistence (Git-style branching)
and GPU memory availability checks.

This module provides the coroutine ``generate_stream_and_persist`` used by
``StreamOllamaClient.create_chat``. It yields ``OllamaStreamingChunk``
objects in real time while accumulating the full assistant message. When
the stream ends, the accumulated data is converted to an ``OllamaResponse``
and persisted **exactly once** via ``PersistenceManager.persist_chat_turn``.

GPU Memory Check
----------------
The streaming HTTP call is wrapped with ``is_ollama_gpu_memory_error`` +
``wrap_ollama_gpu_mem_error``. If Ollama returns a 500 / exception containing
"out of memory", "CUDA out of memory", "failed to allocate", etc., we raise
a typed ``OllamaGPUMemoryWarning`` so callers can react gracefully (e.g.
reduce num_ctx, switch model, fall back to CPU-only Ollama, notify user).

See Also
--------
chat_turn_ollama : non-streaming counterpart (identical GPU check + persistence).
ai_api.core.ollama.errors_ollama : contains the detector and wrapper.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...data_structures.base_objects import SaveMode
from ...data_structures.ollama_objects import (
    OllamaRequest,
    OllamaResponse,
    OllamaStreamingChunk,
)
from ..common.persistence import PersistenceManager
from .errors_ollama import (
    is_ollama_gpu_memory_error,
    wrap_ollama_api_error,
    wrap_ollama_gpu_mem_error,
)


async def generate_stream_and_persist(
    logger: logging.Logger,
    persistence_manager: PersistenceManager | None,
    http_client: httpx.AsyncClient,
    request: OllamaRequest,
    save_mode: str = "none",
    **kwargs: Any,
) -> AsyncIterator[OllamaStreamingChunk]:
    """Stream tokens from Ollama and persist the final response (with GPU memory guard).

    Parameters
    ----------
    logger : logging.Logger
        Structured logger (passed from the client).
    persistence_manager : PersistenceManager or None
        If ``None`` or ``save_mode="none"`` no persistence occurs.
    http_client : httpx.AsyncClient
        Pre-configured client pointing at the Ollama host.
    request : OllamaRequest
        The fully constructed request.
    save_mode : {"none", "json_files", "postgres"}, default "none"
        Passed through to ``persist_chat_turn``.
    **kwargs
        Additional branching parameters forwarded to ``persist_chat_turn``.

    Yields
    ------
    OllamaStreamingChunk
        One chunk per token. The final chunk has ``is_final=True``.

    Raises
    ------
    OllamaGPUMemoryWarning
        If the local GPU runs out of VRAM while loading or running the model
        (detected via exception message patterns).
    httpx.HTTPStatusError
        Propagated (or wrapped) from the streaming POST.
    Exception
        Any persistence error is logged at WARNING but does not abort the
        stream.
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

                if "message" in chunk_data and "content" in chunk_data["message"]:
                    accumulated["message"]["content"] += chunk_data["message"][
                        "content"
                    ]

                if chunk_data.get("done"):
                    accumulated.update(chunk_data)
                    usage = {
                        "total_duration": chunk_data.get("total_duration"),
                        "load_duration": chunk_data.get("load_duration"),
                        "prompt_eval_count": chunk_data.get("prompt_eval_count"),
                        "eval_count": chunk_data.get("eval_count"),
                    }
                    break

                yield OllamaStreamingChunk.from_dict(chunk_data)

        # Build final response and persist
        final_response = OllamaResponse.from_dict(accumulated)
        final_response.usage = usage

        if save_mode != "none" and persistence_manager is not None:
            try:
                await persistence_manager.persist_chat_turn(
                    provider_response=final_response,
                    provider_request=request,
                    kind="chat",
                    branching=True,
                    **kwargs,
                )
            except Exception as exc:
                logger.warning(
                    "Streaming final-response persistence failed",
                    extra={"error": str(exc)},
                )

        yield OllamaStreamingChunk(
            text="",
            finish_reason=accumulated.get("done_reason", "stop"),
            tool_calls_delta=None,
            is_final=True,
            raw=accumulated,
        )

    except Exception as exc:
        # GPU memory availability check (reactive)
        if is_ollama_gpu_memory_error(exc):
            raise wrap_ollama_gpu_mem_error(
                exc,
                f"GPU memory exhausted while streaming chat with model '{request.model}' "
                "(consider reducing num_ctx, using a smaller model, or falling back to CPU)",
            ) from exc

        # Best-effort partial persistence on other errors
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

        # Re-raise (wrapped if it was an API error)
        if isinstance(exc, httpx.HTTPStatusError):
            raise wrap_ollama_api_error(
                exc, f"Ollama streaming chat failed for model '{request.model}'"
            ) from exc
        raise
