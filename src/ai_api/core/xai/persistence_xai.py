"""Postgres persistence layer for xAIClient using infopypg.

Full xAIPersistenceManager class assembled from your provided methods.
Compatible with the Responses API endpoint and your DB schema.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

# infopypg imports
from infopypg import (
    PgPoolManager,
    ensure_partition_exists,
    execute_query,
)

# Local project imports
from ...data_structures.xai_objects import xAIRequest, xAIResponse
from .errors_xai import (
    wrap_infopypg_error,
    xAIInfopypgError,
)

__all__: list[str] = ["xAIPersistenceManager"]

# ------------------------------------------------------------------
# SQL constants
# ------------------------------------------------------------------
SQL_INSERT_REQUEST: str = """
    INSERT INTO requests (
        provider_id, endpoint, request_id, payload, meta
    ) VALUES ($1, $2, $3, $4, $5)
    RETURNING tstamp
"""

SQL_INSERT_RESPONSE: str = """
    INSERT INTO responses (
        provider_id, endpoint, request_id, request_tstamp,
        response_id, payload, meta
    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
"""

SQL_GET_PROVIDER_IDX: str = "SELECT id FROM providers WHERE name = $1"

SQL_ADD_PROVIDER: str = """
    INSERT INTO providers (name, description)
    VALUES ($1, $2)
    ON CONFLICT (name) DO NOTHING
"""

SQL_SELECT_ALL_BATCH_REQUESTS: str = """
    SELECT request_id, tstamp 
    FROM requests 
    WHERE meta->>'batch_id' = $1 
    ORDER BY (meta->>'batch_index')::int ASC
"""


class xAIPersistenceManager:
    """Manages all PostgreSQL persistence for xAI requests/responses.

    Fully compatible with your db_responses_schema.py (partitioned tables,
    composite FKs, provider normalisation, JSONB meta fields).
    """

    def __init__(
        self,
        pg_resolved_settings: Any = None,
        logger: logging.Logger | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self._pg_resolved_settings = pg_resolved_settings
        self._pool: Any = None
        self.logger = logger or logging.getLogger(__name__)
        self.conversation_id = conversation_id or "unknown"

        # ------------------------------------------------------------------
        # Pool & provider helpers (from your original code)
        # ------------------------------------------------------------------

    async def get_pool(self) -> Any:
        if self._pool is None:
            if self._pg_resolved_settings is None or PgPoolManager is None:
                self.logger.error(
                    "Postgres persistence requested without resolved settings"
                )
                raise xAIInfopypgError(
                    "No pg_resolved_settings provided but save_mode=postgres"
                ) from None
            try:
                self._pool = await PgPoolManager.get_pool(self._pg_resolved_settings)
            except Exception as exc:
                self.logger.error(
                    "infopypg pool acquisition failed",
                    extra={"obj": {"error": str(exc)}},
                )
                raise wrap_infopypg_error(
                    exc, "Failed to acquire Postgres pool"
                ) from exc
        return self._pool

    async def _get_pool(self) -> Any:                                                     # alias used in your original methods
        return await self.get_pool()

    async def get_or_create_provider_id(self, name: str = "xai") -> int:
        """Return provider_id for 'xai'; create if missing (normalisation)."""
        pool = await self._get_pool()
        result = await execute_query(
            pool,
            SQL_GET_PROVIDER_IDX,
            params=[name],
            fetch=True,
            logger=self.logger,
        )
        if result and result[0].get("id") is not None:
            return result[0]["id"]

        await execute_query(
            pool,
            SQL_ADD_PROVIDER,
            params=[name, "xai"],
            fetch=False,
            logger=self.logger,
        )
        result = await execute_query(
            pool,
            SQL_GET_PROVIDER_IDX,
            params=[name],
            fetch=True,
            logger=self.logger,
        )
        return result[0]["id"] if result else 1

    def _build_endpoint(self, request: xAIRequest) -> dict[str, Any]:
        """Consistent endpoint metadata – aligned with Responses API."""
        return {
            "provider": "xai",
            "model": request.model,
            "host": "api.x.ai",
            "endpoint_path": "/v1/responses",
            "prompt_cache_key": request.prompt_cache_key,
        }

    # ------------------------------------------------------------------
    # Media placeholder (extend when multimodal support is added)
    # ------------------------------------------------------------------
    async def _save_media_files(
        self, response_id: uuid.UUID, request: xAIRequest
    ) -> list[str]:
        """Placeholder – returns empty list for text-only client."""
        if request.has_media():
            self.logger.info(
                "Media files detected but saving not yet implemented",
                extra={"response_id": str(response_id)},
            )
        return []

    # ------------------------------------------------------------------
    # Persistence methods (your original logic, now class methods)
    # ------------------------------------------------------------------
    async def persist_request(
        self,
        request: xAIRequest,
        batch_id: str | None = None,
        batch_index: int | None = None,
    ) -> tuple[uuid.UUID, datetime]:
        """Persist BEFORE API call."""
        provider_id = await self.get_or_create_provider_id()
        request_id = uuid.uuid4()
        endpoint = self._build_endpoint(request)
        request_payload = asdict(request)

        meta = {
            "conversation_id": self.conversation_id,
            "prompt_snippet": request.extract_prompt_snippet(),
            "prompt_cache_key": request.prompt_cache_key,
            "batch_request_id": request.batch_request_id,
            "batch_id": batch_id,
            "batch_index": batch_index,
            "has_media": request.has_media(),
        }

        today = datetime.now(timezone.utc).date()
        await ensure_partition_exists(
            connection_pool=await self._get_pool(),
            table_name="requests",
            target_date=today,
            logger=self.logger,
        )

        pool = await self._get_pool()
        result = await execute_query(
            pool,
            SQL_INSERT_REQUEST,
            params=[provider_id, endpoint, str(request_id), request_payload, meta],
            fetch=True,
            logger=self.logger,
        )

        tstamp = result[0]["tstamp"] if result else datetime.now(timezone.utc)
        self.logger.info(
            "Request persisted to PostgreSQL",
            extra={
                "obj": {
                    "request_id": str(request_id),
                    "model": request.model,
                    "batch_id": batch_id,
                    "batch_index": batch_index,
                }
            },
        )
        return request_id, tstamp

    async def persist_response(
        self,
        request_id: uuid.UUID,
        request_tstamp: datetime,
        api_result: dict[str, Any],
        request: xAIRequest | None = None,
        batch_id: str | None = None,
    ) -> None:
        """Persist AFTER successful API call."""
        provider_id = await self.get_or_create_provider_id()
        response_id = uuid.uuid4()

        media_files: list[str] = []
        if request is not None:
            media_files = await self._save_media_files(response_id, request)

        endpoint = (
            self._build_endpoint(request)
            if request is not None
            else {
                "provider": "xai",
                "model": api_result.get("model", "unknown"),
                "host": "api.x.ai",
                "endpoint_path": "/v1/responses",
                "prompt_cache_key": None,
            }
        )

        raw_data = api_result.get("raw", api_result)
        if not isinstance(raw_data, dict):
            raw_data = {
                "output": api_result.get("output"),
                "model": api_result.get("model"),
            }

        grok_resp = xAIResponse.from_dict(raw_data)
        response_payload = {
            **raw_data,
            "text": grok_resp.text,
            "tool_calls": grok_resp.tool_calls,
        }

        meta = {
            "conversation_id": self.conversation_id,
            "reasoning_text": grok_resp.reasoning_text,
            "finish_reason": api_result.get("finish_reason"),
            "batch_id": batch_id,
            "media_files": media_files,
        }

        today = datetime.now(timezone.utc).date()
        await ensure_partition_exists(
            connection_pool=await self._get_pool(),
            table_name="responses",
            target_date=today,
            logger=self.logger,
        )

        pool = await self._get_pool()
        await execute_query(
            pool,
            SQL_INSERT_RESPONSE,
            params=[
                provider_id,
                endpoint,
                str(request_id),
                request_tstamp,
                str(response_id),
                response_payload,
                meta,
            ],
            fetch=False,
            logger=self.logger,
        )

        self.logger.info(
            "Response persisted to PostgreSQL",
            extra={
                "obj": {
                    "response_id": str(response_id),
                    "reasoning_captured": bool(grok_resp.reasoning_text),
                    "batch_id": batch_id,
                }
            },
        )

    async def persist_batch_results(
        self, batch_id: str, sdk_results: dict[str, Any]
    ) -> None:
        """Persist succeeded batch responses (your original logic)."""
        # (unchanged – uses your SQL_SELECT_ALL_BATCH_REQUESTS and _persist_response)
        pool = await self._get_pool()
        requests_data = await execute_query(
            pool,
            SQL_SELECT_ALL_BATCH_REQUESTS,
            params=[batch_id],
            fetch=True,
            logger=self.logger,
        )
        if not requests_data:
            self.logger.warning(
                "No requests found for batch", extra={"obj": {"batch_id": batch_id}}
            )
            return

        succeeded = sdk_results.get("succeeded", [])
        persisted_count = 0
        for i, result_item in enumerate(succeeded):
            if i >= len(requests_data):
                break
            req_row = requests_data[i]
            request_id = uuid.UUID(req_row["request_id"])
            request_tstamp = req_row["tstamp"]

            api_result = {
                "raw": result_item
                if isinstance(result_item, dict)
                else {"result": result_item},
                "model": getattr(result_item, "model", "unknown"),
                "finish_reason": getattr(result_item, "finish_reason", None),
            }

            await self.persist_response(
                request_id=request_id,
                request_tstamp=request_tstamp,
                api_result=api_result,
                request=None,
                batch_id=batch_id,
            )
            persisted_count += 1

        self.logger.info(
            "Batch responses persisted",
            extra={"obj": {"batch_id": batch_id, "succeeded_count": persisted_count}},
        )

    async def persist_batch_requests(
        self, batch_id: str, requests: list[xAIRequest]
    ) -> None:
        """Persist every request in a batch."""
        for i, req in enumerate(requests):
            await self.persist_request(req, batch_id=batch_id, batch_index=i)
            self.logger.info(
                "Batch request persisted",
                extra={"obj": {"batch_id": batch_id, "index": i, "model": req.model}},
            )
