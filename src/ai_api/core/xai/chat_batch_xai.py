"""
Example of updated chat_batch_xai.py using the new symmetrical persistence pattern.
"""

from __future__ import annotations

import logging
from typing import Any

from ...data_structures.xai_objects import xAIBatchRequest, xAIBatchResponse
from ..common.persistence import PersistenceManager, persist_batch_requests


async def create_batch_chat(
    client: Any,
    messages_list: list[list[dict]],
    model: str = "grok-4",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> xAIBatchResponse:
    """Batch chat with new symmetrical persistence pattern."""

    logger = client.logger
    logger.info("Creating xAI batch chat", extra={"batch_size": len(messages_list)})

    # Build batch request
    requests = [
        client._build_request(messages, model, temperature, max_tokens, **kwargs)
        for messages in messages_list
    ]
    batch_request = xAIBatchRequest(requests=requests)

    # Persist all requests (new helper)
    if client.persistence_manager is not None:
        try:
            await persist_batch_requests(client.persistence_manager, requests)
        except Exception as exc:
            logger.warning(
                "Batch request persistence failed (continuing)",
                extra={"error": str(exc)},
            )

            # Call the actual batch API (simplified)
    raw_results = await client._call_batch_api(batch_request)

    batch_response = xAIBatchResponse.from_dict(raw_results)

    # Persist the batch response
    if client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_response(
                batch_response, request=batch_request
            )
        except Exception as exc:
            logger.warning(
                "Batch response persistence failed (continuing)",
                extra={"error": str(exc)},
            )

    return batch_response
