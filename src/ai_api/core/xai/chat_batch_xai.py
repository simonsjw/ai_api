"""Batch chat functionality for the xAI API.

Creates a true asynchronous batch job using the official xAI Batch API
(POST /v1/batches → POST /v1/batches/{batch_id}/requests).

Fully integrated with:
- persistence_xai.py (batch-level request + result persistence)
- structured logging (extra={"obj": ...})
- custom error hierarchy
- xAIRequest / xAIBatchRequest data structures
- Responses API payload format inside each batch request
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from ...data_structures.xai_objects import (
    SaveMode,
    xAIBatchRequest,
    xAIBatchResponse,
    xAIInput,
    xAIRequest,
)
from ..xai_client import BaseXAIClient
from .common_xai import _generate_non_streaming
from .errors_xai import wrap_infopypg_error, xAIClientBatchError
from .persistence_xai import xAIPersistenceManager

__all__: list[str] = ["create_batch_chat"]


async def create_batch_chat(
    client: BaseXAIClient,
    messages: list[dict[str, Any]],
    model: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    save_mode: SaveMode = "none",
    batch_name: str | None = None,
    **kwargs: Any,
) -> xAIBatchResponse:
    """Submit a single chat request as part of a new xAI batch job.

    For true multi-request batches, construct an xAIBatchRequest and call
    persist_batch_requests / persist_batch_results manually, or extend this
    function to accept a list of messages lists.

    This function returns an xAIBatchResponse containing the batch_id so you
    can poll /v1/batches/{batch_id} later for results.
    """
    client.logger.info(
        "Creating xAI batch chat",
        extra={
            "obj": {
                "model": model,
                "save_mode": save_mode,
                "batch_name": batch_name,
            }
        },
    )

    # 1. Build canonical xAIRequest (reuses all your existing validation / helpers)
    xai_input = xAIInput.from_list(messages)
    request = xAIRequest(
        input=xai_input,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        save_mode=save_mode,
        **kwargs,
    )

    # 2. Generate deterministic batch identifiers
    batch_id = f"batch-{uuid.uuid4().hex[:12]}"
    batch_name = batch_name or f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    request_id: uuid.UUID | None = None
    request_tstamp: datetime | None = None

    # === PERSISTENCE: BATCH REQUEST (before API call) ===
    if save_mode == "postgres" and client.persistence_manager is not None:
        try:
            # persist_batch_requests expects a list of xAIRequest
            await client.persistence_manager.persist_batch_requests(
                batch_id=batch_id,
                requests=[request],
            )
            client.logger.info(
                "Batch request persisted",
                extra={"obj": {"batch_id": batch_id, "batch_name": batch_name}},
            )
        except Exception as exc:
            client.logger.warning(
                "Batch request persistence failed (continuing)",
                extra={"obj": {"batch_id": batch_id, "error": str(exc)}},
            )
            # still raise so caller knows persistence failed
            raise wrap_infopypg_error(exc, "Failed to persist batch request") from exc

    try:
        # Step 1: Create the batch container
        create_batch_payload = {"name": batch_name}
        batch_create_raw = await _generate_non_streaming(
            client=client,
            endpoint="/v1/batches",
            json_data=create_batch_payload,
        )

        actual_batch_id = batch_create_raw.get("batch_id") or batch_create_raw.get("id")
        if not actual_batch_id:
            raise xAIClientBatchError("xAI Batch creation did not return a batch_id")

        client.logger.info(
            "Batch container created",
            extra={"obj": {"batch_id": actual_batch_id, "batch_name": batch_name}},
        )

        # Step 2: Add the request to the batch (xAI expects Responses API format inside)
        add_requests_payload = {
            "batch_requests": [
                {
                    "batch_request_id": str(uuid.uuid4()),
                    "batch_request": request.to_api_kwargs(),                             # reuses your clean Responses payload
                }
            ]
        }

        add_raw = await _generate_non_streaming(
            client=client,
            endpoint=f"/v1/batches/{actual_batch_id}/requests",
            json_data=add_requests_payload,
        )

        # Build canonical response object
        batch_response = xAIBatchResponse.from_dict(
            {
                "id": actual_batch_id,
                "name": batch_name,
                "created_at": batch_create_raw.get("created_at"),
                "status": "in_progress",
                "results": add_raw,
                "raw": {**batch_create_raw, "added_requests": add_raw},
            }
        )

        client.logger.info(
            "Batch request submitted successfully",
            extra={"obj": {"batch_id": actual_batch_id, "model": model}},
        )

        return batch_response

    except Exception as exc:
        client.logger.error(
            "Batch chat creation failed",
            extra={
                "obj": {"batch_id": batch_id, "batch_name": batch_name, "model": model}
            },
        )
        raise xAIClientBatchError(f"Batch chat creation failed: {exc}") from exc
