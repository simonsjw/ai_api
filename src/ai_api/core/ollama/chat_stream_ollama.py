"""
Example of updated chat_stream_ollama.py using the new symmetrical persistence pattern.

For streaming, we persist the FINAL response object (after accumulating chunks).
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
    """Streaming with new symmetrical persistence pattern."""

    # 1. Create JSON response spec.
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
