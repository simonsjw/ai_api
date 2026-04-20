"""Streaming handler for Ollama native API responses.

Mirrors the exact structure and behaviour of chat_stream_xai.py.
Handles real-time token streaming + optional final persistence.

Key transparent differences (all documented inline):
- Uses httpx.AsyncClient + NDJSON streaming from /api/chat?stream=true
- Each line is a JSON chunk with "message.content" and "done" flag
- OllamaStreamingChunk implements the same LLMStreamingChunkProtocol
- Reuses your existing xAIPersistenceManager unchanged
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...data_structures.ollama_objects import (
    LLMStreamingChunkProtocol,
    OllamaRequest,
    OllamaStreamingChunk,
    SaveMode,
)
from ..common.persistence import PersistenceManager
from .errors_ollama import wrap_ollama_error                                              # new shared Ollama errors

__all__: list[str] = ["ollama_stream", "generate_stream_and_persist"]


async def ollama_stream(
    logger: logging.Logger,
    http_client: "httpx.AsyncClient",                                                     # passed from ollama_client.py
    request: OllamaRequest,
) -> AsyncIterator[LLMStreamingChunkProtocol]:
    """Low-level native Ollama streaming wrapper (mirrors xai_stream)."""
    payload = request.to_ollama_dict()
    payload["stream"] = True

    try:
        async with http_client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk_raw = json.loads(line)

                # Ollama final chunk has "done": true
                is_final = chunk_raw.get("done", False)
                text = (
                    chunk_raw.get("message", {}).get("content", "")
                    if not is_final
                    else ""
                )

                yield OllamaStreamingChunk(
                    text=text,
                    finish_reason=chunk_raw.get("done_reason"),
                    is_final=is_final,
                    done_reason=chunk_raw.get("done_reason"),
                    total_duration=chunk_raw.get("total_duration"),
                    raw={"chunk": chunk_raw},
                )

    except Exception as exc:
        logger.error("Ollama streaming generation failed", extra={"error": str(exc)})
        raise wrap_ollama_error(exc, "Streaming generation failed") from exc


async def generate_stream_and_persist(
    logger: logging.Logger,
    persistence_manager: "PersistenceManager | None",
    http_client: "httpx.AsyncClient",
    request: OllamaRequest,
    save_mode: SaveMode = "none",
) -> AsyncIterator[LLMStreamingChunkProtocol]:
    """Yields chunks in real time + persists final response (mirrors xAI version exactly)."""
    request_id = request_tstamp = None
    if save_mode == "postgres" and persistence_manager is not None:
        try:
            request_id, request_tstamp = await persistence_manager.persist_request(
                request
            )
        except Exception as exc:
            logger.warning(
                "Request persistence failed (continuing)",
                extra={"error": str(exc)},
            )

    full_text: list[str] = []
    final_done_reason: str | None = None

    async for chunk in ollama_stream(logger, http_client, request):
        yield chunk
        if chunk.text:
            full_text.append(chunk.text)
        if getattr(chunk, "is_final", False):
            final_done_reason = getattr(chunk, "finish_reason", None)

            # Final persistence (exact same pattern as xAI)
    if (
        save_mode == "postgres"
        and persistence_manager is not None
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
            "finish_reason": final_done_reason,
            "raw": {"accumulated_text": "".join(full_text)},
        }
        await persistence_manager.persist_response(
            request_id=request_id,
            request_tstamp=request_tstamp,
            api_result=final_result,
            request=request,                                                              # enables media saving
        )
