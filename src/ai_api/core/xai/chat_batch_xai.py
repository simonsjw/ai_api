"""Batch chat functionality for the xAI API using the official xAI SDK.

Now uses the SDK's native `client.batch` sub-client (added in early 2026).
Much cleaner than raw HTTP calls.

Fully integrated with:
- persistence_xai.py (batch-level request + result persistence)
- structured logging
- custom error hierarchy
- xAIRequest / xAIBatchResponse data structures
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

from ...data_structures.xai_objects import (
    SaveMode,
    xAIBatchResponse,
    xAIInput,
    xAIRequest,
)
from ..xai_client import BaseXAIClient
from .errors_xai import wrap_infopypg_error, xAIClientBatchError

__all__: list[str] = ["create_batch_chat", "retrieve_batch_results"]


async def create_batch_chat(
    client: BaseXAIClient,
    messages: list[dict[str, Any]],
    model: str,
    *,
    response_model: type[BaseModel] | None = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    save_mode: SaveMode = "none",
    batch_name: str | None = None,
    **kwargs: Any,
) -> xAIBatchResponse:
    """Submit a single chat request as part of a new xAI batch job using the official SDK.

    Returns an xAIBatchResponse containing the batch_id so you can poll results later.
    """
    client.logger.info(
        "Creating xAI batch chat (via SDK)",
        extra={
            "obj": {
                "model": model,
                "save_mode": save_mode,
                "batch_name": batch_name,
            }
        },
    )

    # 1. Build canonical xAIRequest
    xai_input = xAIInput.from_list(messages)
    request = xAIRequest(
        input=xai_input,
        model=model,
        response_format=response_model,                                                   # xAIRequest uses response_format internally
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        save_mode=save_mode,
        **kwargs,
    )

    # 2. Generate deterministic batch identifiers
    batch_id = f"batch-{uuid.uuid4().hex[:12]}"
    batch_name = batch_name or f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # === PERSISTENCE: BATCH REQUEST (before API call) ===
    if save_mode == "postgres" and client.persistence_manager is not None:
        try:
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
            raise wrap_infopypg_error(exc, "Failed to persist batch request") from exc

    try:
        sdk_client = await client._get_sdk_client()

        # Step 1: Create the batch container via SDK
        batch = await sdk_client.batch.create(batch_name=batch_name)

        actual_batch_id = batch.batch_id
        client.logger.info(
            "Batch container created via SDK",
            extra={"obj": {"batch_id": actual_batch_id, "batch_name": batch_name}},
        )

        # Step 2: Prepare the request as an SDK chat object and add it to the batch
        chat_for_batch = request.prepare_batch_chat(
            sdk_client, batch_request_id=str(uuid.uuid4())
        )

        add_raw = await sdk_client.batch.add(
            batch_id=actual_batch_id,
            batch_requests=[chat_for_batch],
        )

        # Build canonical response object
        batch_response = xAIBatchResponse.from_dict(
            {
                "id": actual_batch_id,
                "name": batch_name,
                "created_at": getattr(batch, "created_at", None),
                "status": "in_progress",
                "results": add_raw,
                "raw": {
                    "batch_create": vars(batch)
                    if hasattr(batch, "__dict__")
                    else batch,
                    "added_requests": add_raw,
                },
            }
        )

        client.logger.info(
            "Batch request submitted successfully via SDK",
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
    response_model: type[BaseModel] | None = None,
) -> xAIBatchResponse:
    """Retrieve a completed xAI batch job using the official SDK."""
    client.logger.info(
        "Retrieving xAI batch results via SDK",
        extra={
            "obj": {
                "batch_id": batch_id,
                "response_model": response_model.__name__ if response_model else None,
            }
        },
    )

    try:
        sdk_client = await client._get_sdk_client()
        batch_raw = await sdk_client.batch.get(batch_id=batch_id)

        # ── Safe conversion to dict (fixes Pyrefly error) ──
        if hasattr(batch_raw, "model_dump"):                                              # Pydantic model
            raw_dict = batch_raw.model_dump()
        elif hasattr(batch_raw, "__dict__"):                                              # normal Python object
            raw_dict = vars(batch_raw)
        else:                                                                             # protobuf object (the common case)
            from google.protobuf.json_format import MessageToDict

            raw_dict = MessageToDict(batch_raw, preserving_proto_field_name=True)

        completed_batch: xAIBatchResponse = xAIBatchResponse.from_dict(raw_dict)

        # ... rest of the function unchanged ...
        client.logger.info(
            "Batch results retrieved successfully via SDK",
            extra={
                "obj": {
                    "batch_id": completed_batch.batch_id,
                    "name": completed_batch.name,
                    "status": completed_batch.status,
                    "result_count": len(completed_batch.results),
                }
            },
        )

        for idx, response in enumerate(completed_batch.results):
            if isinstance(response.parsed, BaseModel):
                client.logger.info(
                    "Processed structured batch result",
                    extra={
                        "obj": {
                            "batch_id": completed_batch.batch_id,
                            "index": idx,
                            "model": type(response.parsed).__name__,
                        }
                    },
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
