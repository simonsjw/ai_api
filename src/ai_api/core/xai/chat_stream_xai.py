"""
Example of updated chat_stream_xai.py using the new symmetrical persistence pattern.

For streaming, we persist the FINAL response object (after accumulating chunks).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from ...data_structures.xai_objects import xAIRequest, xAIResponse, xAIStreamingChunk
from ..common.persistence import PersistenceManager


async def generate_stream_and_persist(
    logger: logging.Logger,
    persistence_manager: PersistenceManager | None,
    chat: Any,                                                                            # xAI SDK chat object
    request: xAIRequest,
    save_mode: str = "none",
) -> AsyncIterator[xAIStreamingChunk]:
    """Streaming with new symmetrical persistence pattern."""

    full_text: list[str] = []
    final_response: xAIResponse | None = None

    async for chunk in chat:                                                              # simplified
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

            # Persist the final response (symmetrical protocol style)
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
