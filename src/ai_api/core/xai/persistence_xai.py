"""Postgres persistence layer for xAIClient using infopypg.

Full xAIPersistenceManager class assembled from your provided methods.
Compatible with the Responses API endpoint and your DB schema.
Now includes full multimodal media persistence.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import urllib.request
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError

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
# SQL constants (unchanged)
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

    Now supports multimodal media persistence when media_root is supplied.
    """

    def __init__(
        self,
        pg_resolved_settings: Any = None,
        logger: logging.Logger | None = None,
        conversation_id: str | None = None,
        media_root: Path | str | None = None,                                             # NEW: required for media saving
    ) -> None:
        self._pg_resolved_settings = pg_resolved_settings
        self._pool: Any = None
        self.logger = logger or logging.getLogger(__name__)
        self.conversation_id = conversation_id or "unknown"
        self.media_root = Path(media_root) if media_root else None

        # ------------------------------------------------------------------
        # Pool & provider helpers (unchanged)
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

    async def _get_pool(self) -> Any:
        return await self.get_pool()

    async def get_or_create_provider_id(self, name: str = "xai") -> int:
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
        return {
            "provider": "xai",
            "model": request.model,
            "host": "api.x.ai",
            "endpoint_path": "/v1/responses",
            "prompt_cache_key": request.prompt_cache_key,
        }

    # ------------------------------------------------------------------
    # FULL MULTIMODAL MEDIA SAVING (merged from former media_xai.py)
    # ------------------------------------------------------------------
    async def _save_media_files(
        self, response_id: uuid.UUID, request: xAIRequest
    ) -> list[str]:
        """Save multimodal images/files to the monthly/response_id folder structure.

        Returns list of relative paths (from media root) for storage in responses.meta.
        URLs are downloaded; local paths are copied. Index.txt is updated for auditing.
        """
        if not request.has_media() or self.media_root is None:
            return []

        prompt_snippet = request.extract_prompt_snippet(max_chars=100)

        media_items: list[dict[str, str]] = []
        for msg in request.get_messages():
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype in ("input_image", "input_file"):
                            url_or_path = (
                                part.get("image_url")
                                or part.get("file_url")
                                or part.get("url")
                            )
                            if url_or_path:
                                media_items.append(
                                    {
                                        "type": ptype,
                                        "url_or_path": url_or_path,
                                        "original_name": Path(url_or_path).name
                                        or "file",
                                    }
                                )

        if not media_items:
            return []

        month = datetime.now(timezone.utc).strftime("%Y-%m")
        response_folder = self.media_root / month / str(response_id)
        response_folder.mkdir(parents=True, exist_ok=True)

        relative_paths: list[str] = []
        for item in media_items:
            src = item["url_or_path"]
            safe_name = Path(item["original_name"]).name
            dest_path = response_folder / safe_name

            try:
                if src.startswith(("http://", "https://")):

                    def _download() -> None:
                        urllib.request.urlretrieve(src, str(dest_path))

                    await asyncio.to_thread(_download)
                else:
                    await asyncio.to_thread(shutil.copy2, src, dest_path)

                self.logger.info(
                    "Media file saved successfully",
                    extra={"obj": {"response_id": str(response_id), "src": src}},
                )
            except (URLError, OSError, FileNotFoundError) as exc:
                self.logger.warning(
                    "Failed to save media file",
                    extra={
                        "obj": {
                            "response_id": str(response_id),
                            "src": src,
                            "error": str(exc),
                        }
                    },
                )
                continue

            relative_path = f"{month}/{response_id}/{safe_name}"
            relative_paths.append(relative_path)

            # Append to index.txt
            index_line = (
                f"{response_id}|"
                f"{datetime.now(timezone.utc).isoformat()}|"
                f"{relative_path}|"
                f"{prompt_snippet}\n"
            )
            index_path = self.media_root / "index.txt"

            def _append_index() -> None:
                index_path.parent.mkdir(parents=True, exist_ok=True)
                with index_path.open("a", encoding="utf-8") as f:
                    f.write(index_line)

            await asyncio.to_thread(_append_index)

        return relative_paths

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
                    "media_files_count": len(media_files),
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
