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
    xAIJSONResponseSpec,
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
    response_model: type["xAIJSONResponseSpec"] | None = None,                            # <-- NEW
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
        response_model=response_model,
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


async def retrieve_batch_results(
    client: BaseXAIClient,
    batch_id: str,
    response_model: type["xAIJSONResponseSpec"] | None = None,
) -> xAIBatchResponse:
    """Retrieve a completed xAI batch job and return a fully wrapped xAIBatchResponse.

    Each item in .results is an xAIResponse instance, so .parsed (structured output)
    is available immediately when response_model was supplied at creation time.
    """
    client.logger.info(
        "Retrieving xAI batch results",
        extra={
            "obj": {
                "batch_id": batch_id,
                "response_model": response_model.__name__ if response_model else None,
            }
        },
    )

    try:
        # Fetch raw batch status and results via the existing non-streaming helper
        batch_raw = await _generate_non_streaming(
            client=client,
            endpoint=f"/v1/batches/{batch_id}",                                           # no json_data since GET requests send no body.
        )

        # Use the canonical factory – automatically wraps every result
        # using xAIResponse.from_dict / from_sdk
        completed_batch: xAIBatchResponse = xAIBatchResponse.from_dict(batch_raw)

        # Optional: update status in the wrapper if the API reports it
        if completed_batch.status is None:
            completed_batch = completed_batch.model_copy(
                update={"status": batch_raw.get("status")}
            )

        client.logger.info(
            "Batch results retrieved successfully",
            extra={
                "obj": {
                    "batch_id": completed_batch.batch_id,
                    "name": completed_batch.name,
                    "status": completed_batch.status,
                    "result_count": len(completed_batch.results),
                }
            },
        )

        # === Structured output handling (validated_model = response.parsed) ===
        for idx, response in enumerate(completed_batch.results):
            if response.parsed is not None:
                validated_model: "xAIJSONResponseSpec" = response.parsed
                client.logger.info(
                    "Processed structured batch result",
                    extra={
                        "obj": {
                            "batch_id": completed_batch.batch_id,
                            "index": idx,
                            "model": type(validated_model).__name__,
                        }
                    },
                )
                # Example: persist the validated model (extend persistence_xai.py as needed)
                # await client.persistence_manager._persist_validated_model(
                #     batch_id=completed_batch.batch_id, index=idx, model=validated_model
                # )
            else:
                # Non-structured or fallback result
                client.logger.debug(
                    "Non-structured batch result",
                    extra={"obj": {"batch_id": completed_batch.batch_id, "index": idx}},
                )

        return completed_batch

    except Exception as exc:
        client.logger.error(
            "Batch results retrieval failed",
            extra={"obj": {"batch_id": batch_id}},
        )
        raise xAIClientBatchError(
            f"Failed to retrieve batch {batch_id}: {exc}"
        ) from exc
