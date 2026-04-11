"""Batch processor for xAI API batch operations.

This module defines the ``xAIBatchProcessor`` class, which is responsible
for all batch lifecycle operations with the xAI Responses API.

It handles:
- Creation of named batches
- Addition of one or more ``xAIRequest`` objects (including full multimodal
  support via text, images and files)
- Retrieval of batch status and results
- Coordination of persistence for batch requests and successful responses
  when ``save_mode`` is set to 'postgres'

All operations are performed asynchronously, preserve structured logging,
and translate SDK exceptions into domain-specific errors.
"""

from __future__ import annotations

from typing import Any

# xAI SDK imports (message construction helpers)
from xai_sdk.chat import file, image, system, user

# Local project imports
from ...data_structures.xai_objects import *
from .xai_errors import (
    xAIClientBatchError,
    xAIClientMultimodalError,
)

__all__: list[str] = [
    "create_batch",
    "add_to_batch",
    "get_batch_status",
    "retrieve_batch_results",
    "retrieve_and_persist_batch_results",
]


async def create_batch(self, batch_name: str) -> dict[str, Any]:
    try:
        batch = await self._client.batch.create(batch_name=batch_name)                    # type: ignore[attr-defined]
        self.logger.info("Batch created", extra={"obj": {"batch_name": batch_name}})
        return {
            "batch_id": getattr(batch, "batch_id", None),
            "batch_name": batch_name,
        }
    except Exception as exc:
        self.logger.error(
            "Failed to create batch",
            extra={"obj": {"batch_name": batch_name, "error": str(exc)}},
        )
        raise xAIClientBatchError("Failed to create batch") from exc


async def add_to_batch(self, batch_id: str, requests: list[xAIRequest]) -> None:
    if not batch_id:
        self.logger.error("add_to_batch called with empty batch_id")
        raise xAIClientBatchError("batch_id is required") from None
    try:
        batch_requests = []
        for req in requests:
            chat = self._client.chat.create(
                model=req.model,
                batch_request_id=req.batch_request_id,                                    # type: ignore[call-arg]
                store_messages=True,                                                      # Enables modern Responses API
            )
            for msg_dict in req.to_sdk_messages():
                role = msg_dict["role"]
                content = msg_dict["content"]
                if role == "system":
                    chat.append(system(content))
                elif role == "user":
                    if isinstance(content, str):
                        chat.append(user(content))
                    elif isinstance(content, list):
                        # Multimodal validation & handling
                        parts = []
                        for part in content:
                            if part.get("type") == "input_text":
                                parts.append(part["text"])
                            elif part.get("type") == "input_image":
                                parts.append(image(part["image_url"]))
                            elif part.get("type") == "input_file":
                                # Native file support (public URL or uploaded file_id)
                                parts.append(file(part["file_url"]))                      # placeholder
                            else:
                                self.logger.error(
                                    "Unsupported multimodal content type",
                                    extra={"obj": {"part": part}},
                                )
                                raise xAIClientMultimodalError(
                                    "Unsupported content type in multimodal message"
                                ) from None
                        chat.append(user(*parts))
                else:
                    chat.append(user(content))

            batch_requests.append(chat)

            # capture the batch.
            # TODO: ensure that if not captured here, the server is set to capture?
        if any(req.save_mode == "postgres" for req in requests):
            await self._persist_batch_requests(batch_id, requests)
        else:
            self.logger.info(
                "Batch persistence skipped (save_mode != 'postgres')"
            )                                                                             # 90-col comment start

        await self._client.batch.add(                                                     # type: ignore[attr-defined]
            batch_id=batch_id, batch_requests=batch_requests
        )
        self.logger.info(
            "Requests added to batch",
            extra={"obj": {"batch_id": batch_id, "count": len(requests)}},
        )

    except Exception as exc:
        self.logger.error(
            "Failed to add requests to batch",
            extra={"obj": {"batch_id": batch_id, "error": str(exc)}},
        )
        raise xAIClientBatchError("Failed to add requests to batch") from exc


async def get_batch_status(self, batch_id: str) -> dict[str, Any]:
    try:
        batch = await self._client.batch.get(batch_id=batch_id)                           # type: ignore[attr-defined]
        return {
            "batch_id": batch_id,
            "state": getattr(batch, "state", None),
        }
    except Exception as exc:
        self.logger.error(
            "Failed to get batch status",
            extra={"obj": {"batch_id": batch_id, "error": str(exc)}},
        )
        raise xAIClientBatchError("Failed to get batch status") from exc


async def retrieve_batch_results(
    self, batch_id: str, limit: int = 100
) -> dict[str, Any]:
    try:
        results = await self._client.batch.list_batch_results(                            # type: ignore[attr-defined]
            batch_id=batch_id, limit=limit
        )
        return {
            "succeeded": getattr(results, "succeeded", []),
            "failed": getattr(results, "failed", []),
            "pagination_token": getattr(results, "pagination_token", None),
        }
    except Exception as exc:
        self.logger.error(
            "Failed to retrieve batch results",
            extra={"obj": {"batch_id": batch_id, "error": str(exc)}},
        )
        raise xAIClientBatchError("Failed to retrieve batch results") from exc


async def retrieve_and_persist_batch_results(
    self, batch_id: str, limit: int = 100
) -> dict[str, Any]:
    """Retrieve batch results from xAI SDK and automatically persist all succeeded responses.
    All responses will carry the same batch_id in meta for easy identification/grouping."""
    results = await self.retrieve_batch_results(batch_id, limit)

    if results.get("succeeded"):
        await self._persist_batch_results(batch_id, results)

    return results
