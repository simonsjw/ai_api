"""Postgres persistence layer for xAIClient using infopypg.

Full xAIPersistenceManager class assembled from provided methods in objects.
Compatible with the Responses API endpoint and DB schema.
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
# SQL constants (original + new branching queries)
# ------------------------------------------------------------------
SQL_INSERT_REQUEST: str = """
    INSERT INTO requests (
        provider_id, endpoint, request_id, payload, meta,
        tree_id, branch_id, parent_response_id, sequence
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    RETURNING tstamp
"""

SQL_INSERT_RESPONSE: str = """
    INSERT INTO responses (
        provider_id, endpoint, request_id, request_tstamp,
        response_id, payload, meta,
        tree_id, branch_id, parent_response_id, sequence
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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

# ─────────────────────────────────────────────────────────────────────────────
# Conversation + Branching SQL
# ─────────────────────────────────────────────────────────────────────────────
SQL_ENSURE_CONVERSATION: str = """
    INSERT INTO conversations (tree_id, title, meta)
    VALUES ($1, $2, $3)
    ON CONFLICT (tree_id) DO NOTHING
"""

SQL_SAVE_TO_HISTORY: str = """
    INSERT INTO messages (
        message_id, tree_id, branch_id, parent_message_id,
        sequence, role, content, request_id, response_id, meta
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
"""

SQL_LOAD_BRANCH_HISTORY: str = """
    WITH RECURSIVE branch_chain AS (
        SELECT * FROM responses
        WHERE response_id = $1
        UNION ALL
        SELECT r.* FROM responses r
        INNER JOIN branch_chain bc ON r.response_id = bc.parent_response_id
    )
    SELECT * FROM branch_chain
    ORDER BY sequence DESC
"""

SQL_CREATE_BRANCH: str = """
    INSERT INTO responses (
        ... -- same columns as INSERT_RESPONSE but with new branch_id
    ) VALUES (...)  -- implemented in Python for clarity
"""

SQL_LIST_BRANCHES: str = """
    SELECT DISTINCT branch_id, MIN(sequence) as first_sequence,
           COUNT(*) as message_count, MAX(tstamp) as last_activity
    FROM responses
    WHERE tree_id = $1
    GROUP BY branch_id
    ORDER BY last_activity DESC
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
        media_root: Path | str | None = None,                                             # required for media saving
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
    # Conversation & Branching Helpers
    # ------------------------------------------------------------------
    async def ensure_conversation_exists(
        self, tree_id: uuid.UUID, title: str | None = None
    ) -> uuid.UUID:
        """Create conversation tree metadata if it does not exist."""
        pool = await self._get_pool()
        await execute_query(
            pool,
            SQL_ENSURE_CONVERSATION,
            params=[tree_id, title, {}],
            fetch=False,
            logger=self.logger,
        )
        return tree_id

    async def save_to_history(
        self,
        request: xAIRequest,
        response: xAIResponse | None = None,
        tree_id: uuid.UUID | None = None,
        branch_id: uuid.UUID | None = None,
        parent_response_id: uuid.UUID | None = None,
        parent_message_id: uuid.UUID | None = None,
        sequence: int | None = None,
    ) -> dict[str, Any]:
        """Save a turn to the conversation history (Git-style branching).

        This is the main integration point called automatically from persist_response.
        """
        if tree_id is None or branch_id is None:
            return {}

        await self.ensure_conversation_exists(tree_id)

        message_id = uuid.uuid4()
        role = "user" if request else "assistant"

        pool = await self._get_pool()
        await execute_query(
            pool,
            SQL_SAVE_TO_HISTORY,
            params=[
                message_id,
                tree_id,
                branch_id,
                parent_response_id,
                parent_message_id,
                sequence,
                role,
                asdict(request)
                if request is not None
                else asdict(response)
                if response is not None
                else {},
                getattr(request, "request_id", None),
                getattr(response, "id", None) if response else None,
                {"has_media": request.has_media() if request else False},
            ],
            fetch=False,
            logger=self.logger,
        )

        self.logger.info(
            "Message saved to conversation history",
            extra={
                "obj": {
                    "tree_id": str(tree_id),
                    "branch_id": str(branch_id),
                    "sequence": sequence,
                    "parent_response_id": str(parent_response_id)
                    if parent_response_id
                    else None,
                }
            },
        )
        return {"message_id": message_id, "tree_id": tree_id, "branch_id": branch_id}

    async def load_branch_history(
        self,
        tree_id: uuid.UUID,
        branch_id: uuid.UUID | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Reconstruct full linear history for a branch (or latest branch if branch_id omitted)."""
        pool = await self._get_pool()
        query = """
            SELECT response_id, tree_id, branch_id, parent_response_id, sequence,
                   payload, meta, tstamp
            FROM responses
            WHERE tree_id = $1
              AND (branch_id = $2 OR $2 IS NULL)
            ORDER BY sequence DESC
            LIMIT $3
        """
        result = await execute_query(
            pool,
            query,
            params=[tree_id, branch_id, limit],
            fetch=True,
            logger=self.logger,
        )
        return result

    async def create_branch(
        self,
        tree_id: uuid.UUID,
        from_response_id: uuid.UUID,
        new_branch_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """Create a new branch forking from an existing response."""
        if new_branch_id is None:
            new_branch_id = uuid.uuid4()

            # The actual forking logic is handled at the application level by passing
            # parent_response_id when creating the next response.
        self.logger.info(
            "New branch created",
            extra={
                "obj": {"tree_id": str(tree_id), "new_branch_id": str(new_branch_id)}
            },
        )
        return new_branch_id

    async def list_branches(self, tree_id: uuid.UUID) -> list[dict[str, Any]] | None:
        """Return all active branches for a conversation tree."""
        pool = await self._get_pool()
        result = await execute_query(
            pool,
            SQL_LIST_BRANCHES,
            params=[tree_id],
            fetch=True,
            logger=self.logger,
        )
        return result

    # ------------------------------------------------------------------
    # Persistence methods (original logic, now class methods)
    # ------------------------------------------------------------------
    async def persist_request(
        self,
        request: xAIRequest,
        batch_id: str | None = None,
        batch_index: int | None = None,
        tree_id: uuid.UUID | None = None,
        branch_id: uuid.UUID | None = None,
        parent_response_id: uuid.UUID | None = None,
        sequence: int | None = None,
    ) -> tuple[uuid.UUID, datetime]:
        """Persist BEFORE API call – now includes branching metadata."""
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
            "tree_id": str(tree_id) if tree_id else None,
            "branch_id": str(branch_id) if branch_id else None,
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
            params=[
                provider_id,
                endpoint,
                str(request_id),
                request_payload,
                meta,
                tree_id,
                branch_id,
                parent_response_id,
                sequence,
            ],
            fetch=True,
            logger=self.logger,
        )

        tstamp = result[0]["tstamp"] if result else datetime.now(timezone.utc)
        self.logger.info(
            "Request persisted to PostgreSQL",
            extra={"obj": {"request_id": str(request_id), "tree_id": str(tree_id)}},
        )
        return request_id, tstamp

    async def persist_response(
        self,
        request_id: uuid.UUID,
        request_tstamp: datetime,
        api_result: dict[str, Any],
        request: xAIRequest | None = None,
        batch_id: str | None = None,
        tree_id: uuid.UUID | None = None,
        branch_id: uuid.UUID | None = None,
        parent_response_id: uuid.UUID | None = None,
        sequence: int | None = None,
    ) -> None:
        """Persist AFTER successful API call – includes media + history."""
        provider_id = await self.get_or_create_provider_id()
        response_id = uuid.uuid4()

        media_files: list[str] = []
        if request is not None:
            media_files = await self._save_media_files(response_id, request)

            # Auto-save to conversation history if branching info is provided
        if tree_id and branch_id and request is not None:
            await self.save_to_history(
                request=request,
                response=xAIResponse.from_dict(api_result),
                tree_id=tree_id,
                branch_id=branch_id,
                parent_response_id=parent_response_id,
                sequence=sequence,
            )

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
            "tree_id": str(tree_id) if tree_id else None,
            "branch_id": str(branch_id) if branch_id else None,
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
                tree_id,
                branch_id,
                parent_response_id,
                sequence,
            ],
            fetch=False,
            logger=self.logger,
        )

        self.logger.info(
            "Response persisted to PostgreSQL",
            extra={
                "obj": {
                    "response_id": str(response_id),
                    "tree_id": str(tree_id),
                    "branch_id": str(branch_id),
                    "media_files_count": len(media_files),
                }
            },
        )

    async def persist_batch_results(
        self, batch_id: str, sdk_results: dict[str, Any]
    ) -> None:
        """Persist succeeded batch responses (original logic)."""
        # (unchanged – uses SQL_SELECT_ALL_BATCH_REQUESTS and _persist_response)
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
