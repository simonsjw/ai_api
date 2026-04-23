"""
Ollama streaming chat implementation with symmetrical persistence.

This module implements real-time token streaming for Ollama using the native
``/api/chat?stream=true`` endpoint. It is the concrete implementation behind
``StreamOllamaClient.create_chat(...)``.

High-level view of Ollama streaming
-----------------------------------
- Uses httpx streaming response (``async with http_client.stream(...)``).
- Yields ``OllamaStreamingChunk`` objects in real time (text deltas, finish_reason, telemetry).
- After the stream ends, assembles a final ``OllamaResponse`` and persists it
  (the request was already persisted before streaming started).
- Structured output (``response_model``) is supported: the final accumulated
  text is validated against the Pydantic model and attached as ``.parsed``.

Comparison with xAI streaming
-----------------------------
- Ollama: native HTTP streaming, fine-grained telemetry in every chunk
  (total_duration, load_duration, etc.), local execution.
- xAI: SDK-based async iterator, final response built from accumulated chunks,
  richer error types (rate-limit, thinking mode), native batch support in a
  separate module.

The streaming path deliberately re-uses the same persistence and structured-output
machinery as the turn-based path for consistency.

See Also
--------
ai_api.core.ollama_client.StreamOllamaClient
ai_api.core.ollama.chat_turn_ollama
    The non-streaming counterpart (shares the same persistence pattern).
ai_api.core.xai.chat_stream_xai
    The xAI streaming implementation.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Type

from pydantic import BaseModel

from ...data_structures.ollama_objects import (
    OllamaRequest,
    OllamaResponse,
    OllamaStreamingChunk,
)
from ..common.persistence import PersistenceManager
from ..common.response_struct import create_json_response_spec


async def generate_stream_and_persist(
    logger: logging.Logger,
    persistence_manager: PersistenceManager | None,
    http_client: Any,
    request: OllamaRequest,
    save_mode: str = "none",
    response_model: Type[BaseModel] | None = None,
) -> AsyncIterator[OllamaStreamingChunk]:
    """Generate a streaming chat response from Ollama and persist the final result.

    This generator yields chunks as they arrive from Ollama, then (after the
    stream completes) builds and persists a final ``OllamaResponse`` object.
    Structured output validation happens on the accumulated text before
    persistence.

    Parameters
    ----------
    logger : logging.Logger
        Logger for info/warning messages.
    persistence_manager : PersistenceManager or None
        Used to persist the final response (if ``save_mode != "none"``).
    http_client : httpx.AsyncClient
        Pre-configured client pointing at the Ollama host.
    request : OllamaRequest
        The request object (already contains model, messages, generation params,
        save_mode, and optionally response_format).
    save_mode : {"none", "json_files", "postgres"}, optional
        Persistence mode (default "none").
    response_model : type[pydantic.BaseModel] or None, optional
        If supplied, the final text is validated against this model and the
        resulting instance is attached to the final response as ``.parsed``.

    Yields
    ------
    OllamaStreamingChunk
        Real-time chunks containing ``text``, ``finish_reason``, ``is_final``,
        and raw telemetry.

    Notes
    -----
    The request itself should be persisted by the caller (e.g.
    ``StreamOllamaClient``) before calling this function. Only the final
    response is persisted here.
    """

    # 1. Create JSON response spec (note: in current code this incorrectly uses "xai";
    #    it should be "ollama" — documented as-is per instruction to assume code correct).
    if response_model is not None:
        spec = create_json_response_spec("xai", response_model)
        request = request.model_copy(
            update={"response_format": spec.to_sdk_response_format()}
        )

    full_text: list[str] = []
    final_response: OllamaResponse | None = None

    payload = request.to_ollama_dict()
    payload["stream"] = True

    # 2. Collect streaming chunks.
    async with http_client.stream("POST", "/api/chat", json=payload) as resp:
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            chunk_raw = __import__("json").loads(line)

            is_final = chunk_raw.get("done", False)
            text = (
                chunk_raw.get("message", {}).get("content", "") if not is_final else ""
            )

            chunk = OllamaStreamingChunk(
                text=text,
                finish_reason=chunk_raw.get("done_reason"),
                is_final=is_final,
                done_reason=chunk_raw.get("done_reason"),
                total_duration=chunk_raw.get("total_duration"),
                raw={"chunk": chunk_raw},
            )
            yield chunk

            if chunk.text:
                full_text.append(chunk.text)
            if is_final:
                final_response = OllamaResponse.from_dict(chunk_raw)

                # 3. validate with response specification if provided.
    if response_model is not None and final_response is not None:
        try:
            parsed = response_model.model_validate_json("".join(full_text))
            final_response.parsed = parsed
        except Exception as exc:
            logger.warning(
                "Failed to parse final structured chunk", extra={"error": str(exc)}
            )

            # 4. Persist the final response
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
