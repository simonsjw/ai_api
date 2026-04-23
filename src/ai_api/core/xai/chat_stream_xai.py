"""
Example of updated chat_stream_xai.py using the new symmetrical persistence pattern.

For streaming, we persist the FINAL response object (after accumulating chunks).
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
    """Streaming with new symmetrical persistence pattern."""

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
