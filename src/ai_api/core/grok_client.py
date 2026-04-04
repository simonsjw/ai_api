#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GrokClient – native xAI SDK client with Responses API, infopypg persistence,
logger, multimodal/file support, prompt caching, batch processing, and
comprehensive error handling via grok_error.py.

This module provides the `GrokClient` class, a complete wrapper around the
official xAI AsyncClient. It uses the modern stateful Responses API
(/v1/responses endpoint) via store_messages=True, enabling server-side message
persistence and improved reasoning handling. Prompt caching is optimised
through x-grok-conv-id gRPC metadata (set via conversation_id), which routes
requests to the same backend server for maximum cache-hit rates.

Key functionality categories
----------------------------
- **Single generation** (`generate`): text, multimodal, or reasoning-enabled
  calls using the modern Responses API with optional PostgreSQL persistence.
- **Streaming generation** (`_generate_stream_and_persist`): real-time chunk
  yielding combined with final accumulated response persistence.
- **Batch processing** (`create_batch`, `add_to_batch`,
  `retrieve_and_persist_batch_results`): full lifecycle support with per-item
  request/response rows linked by `batch_id`.
- **Database persistence**: requests are saved before the API call;
  responses (including reasoning traces) are saved after success. All rows
  use composite foreign keys and JSONB metadata for efficient querying.
- **Prompt caching**: automatic via x-grok-conv-id metadata derived from
  conversation_id (recommended to match GrokRequest.prompt_cache_key).

Main class
----------
GrokClient
    The primary public interface. All public methods are asynchronous.

Important instance attributes
-----------------------------
- `api_key` (str): xAI API key (from argument or XAI_API_KEY environment
  variable).
- `logger` (Logger): mandatory structured logger for every operation.
- `conversation_id` (str | None): used as x-grok-conv-id for prompt caching
  and server affinity.
- `_client` (xai_sdk.AsyncClient): underlying SDK client configured for
  Responses API.
- `_pool` (Any | None): lazy PostgreSQL connection pool from infopypg.

Public methods and flow
-----------------------
1. `create_request(**data)` → GrokRequest
   Converts a generic dictionary into a fully validated GrokRequest object.

2. `generate(request, stream=False)` → dict | AsyncIterator
   - Persists request (if `save_mode == "postgres"`).
   - Builds and sends messages using the modern Responses API.
   - For non-streaming: calls `_grok_generate`, persists response.
   - For streaming: calls `_generate_stream_and_persist` (accumulates text
     and persists one final row on completion).

3. `create_batch`, `add_to_batch`, `get_batch_status`,
   `retrieve_and_persist_batch_results`
   - Batch lifecycle with automatic per-item request/response persistence
     using `batch_id` and `batch_index` for ordering and identification.

Persistence flow (when `save_mode == "postgres"`)
-------------------------------------------------
- `_persist_request` → inserts into `requests` table (returns `request_id`,
  `tstamp`).
- `_persist_response` → inserts into `responses` table with composite FK to
  the original request and optional `batch_id`.
- `_persist_batch_requests` / `_persist_batch_results` handle bulk cases
  using `batch_id` in JSONB metadata for easy grouping.

Error handling
--------------
Every path logs a structured message via `self.logger` before raising a
`Grok*Error` subclass (or wrapped SDK exception).

Notes
-----
- All database operations use `infopypg` helpers (`execute_query`,
  `ensure_partition_exists`) and respect daily RANGE partitioning.
- Reasoning content is automatically extracted via `GrokResponse.from_dict`
  when `include_reasoning=True`.
- The module is fully type-annotated for Python 3.12 and passes Pyrefly
  static checking.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import urllib.request
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.error import URLError

from infopypg import (
    PgPoolManager,
    ResolvedSettingsDict,
    ensure_partition_exists,
    execute_query,
)
from logger import Logger
from xai_sdk import AsyncClient
from xai_sdk.chat import file, image, system, user                                        # file helper if exposed by SDK

from ..data_structures.grok import (
    GrokBatchRequest,
    GrokRequest,
    GrokResponse,
    GrokStreamingChunk,
    LLMStreamingChunkProtocol,
    SaveMode,
)
from .grok_error import (
    GrokAPIAuthenticationError,
    GrokAPIConnectionError,
    GrokAPIError,
    GrokAPIRateLimitError,
    GrokClientBatchError,
    GrokClientError,
    GrokClientMultimodalError,
    GrokInfopypgError,
    GrokPostgresError,
    UnsupportedThinkingModeError,
    wrap_grok_api_error,
    wrap_infopypg_error,
    wrap_postgres_error,
)

REASONING_MODELS: set[str] = {"grok-4.20-reasoning", "grok-4", "grok-beta"}

SQL_INSERT_REQUEST = """
    INSERT INTO requests (provider_id, endpoint, request_id, request, meta)
    VALUES ($1, $2, $3, $4, $5)
    RETURNING tstamp
"""

SQL_INSERT_RESPONSE = """
    INSERT INTO responses (provider_id, endpoint, request_id, request_tstamp,
                           response_id, response, meta)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
"""


class GrokClient:
    """Dedicated Grok client with full error + logging integration."""

    def __init__(
        self,
        logger: Logger,
        api_key: str | None = None,
        pg_resolved_settings: ResolvedSettingsDict | None = None,
        settings: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        media_root: str = "media",
    ) -> None:
        """Initialise with mandatory logger and optional Postgres settings."""
        try:
            self.api_key = api_key or os.getenv("XAI_API_KEY")
            if not self.api_key:
                if logger:
                    logger.error(
                        "Missing XAI API key",
                        extra={"obj": {"conversation_id": conversation_id}},
                    )
                raise GrokClientError("api_key is required for GrokClient") from None

            self.logger = logger
            self.settings = settings or {}
            self.conversation_id = conversation_id

            # ------------------------------------------------------------------
            # Modern Responses API + prompt-caching optimisation.
            # x-grok-conv-id metadata routes requests to the same server,
            # maximising cache hits. We use conversation_id (or the caller's
            # prompt_cache_key value) as the affinity key.
            # ------------------------------------------------------------------
            metadata: tuple[tuple[str, str], ...] | None = None
            if self.conversation_id:
                metadata = (("x-grok-conv-id", self.conversation_id),)

            self._client = AsyncClient(
                api_key=self.api_key,
                metadata=metadata,
            )

            # ----------------------------------------------------------------
            # Persistence: Postgres & on disk storage.
            # ----------------------------------------------------------------
            self._pg_resolved_settings = pg_resolved_settings
            self._pool = None
            self.media_root: Path = Path(media_root).resolve()
            self.logger.info(
                "GrokClient initialised successfully",
                extra={
                    "obj": {
                        "conversation_id": conversation_id,
                        "media_root": str(self.media_root),
                    }
                },
            )

        except Exception as exc:                                                          # Defensive top-level catch
            if logger:
                logger.error(
                    "GrokClient initialisation failed",
                    extra={"obj": {"error": str(exc)}},
                )
            raise GrokClientError("Failed to initialise GrokClient") from exc

    def create_request(self, **data: Any) -> GrokRequest:
        """Convert generic dict → native GrokRequest (wraps data-layer ValueError)."""
        try:
            return GrokRequest.from_dict(data)
        except ValueError as exc:
            self.logger.error(
                "Invalid request data provided to create_request",
                extra={"obj": {"data_keys": list(data.keys()), "error": str(exc)}},
            )
            raise GrokClientError("Invalid request data") from exc

    async def generate(
        self, request: GrokRequest, stream: bool = False, **kwargs: Any
    ) -> dict[str, Any] | AsyncIterator[LLMStreamingChunkProtocol]:
        """Main generation path with full error handling and pre-raise logging."""
        if request.include_reasoning and request.model not in REASONING_MODELS:
            self.logger.error(
                "Unsupported thinking mode requested",
                extra={"obj": {"model": request.model, "include_reasoning": True}},
            )
            raise UnsupportedThinkingModeError(
                f"Reasoning mode not supported for model {request.model}"
            ) from None

        self.logger.info(
            "Grok generation started",
            extra={
                "obj": {
                    "model": request.model,
                    "stream": stream,
                    "cache_key": request.prompt_cache_key,
                }
            },
        )

        persist_info: dict | None = None
        if request.save_mode == "postgres":
            request_id, request_tstamp = await self._persist_request(request)
            persist_info = {"request_id": request_id, "request_tstamp": request_tstamp}

        try:
            # chat = self._client.chat.create(model=request.model)
            chat = self._client.chat.create(
                model=request.model,
                store_messages=True,                                                      # Enables modern Responses API                             # enables automatic prompt caching
            )
            for msg_dict in request.to_sdk_messages():
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
                                # Extend with SDK file() when available
                                parts.append(image(part.get("file_url")))                 # placeholder
                            else:
                                self.logger.error(
                                    "Unsupported multimodal content type",
                                    extra={"obj": {"part": part}},
                                )
                                raise GrokClientMultimodalError(
                                    "Unsupported content type in multimodal message"
                                ) from None
                        chat.append(user(*parts))
                else:
                    chat.append(user(content))

                    # Here the message is sent to the method that contacts grok.
                    # if stream:
            if stream:
                if persist_info:
                    result = self._generate_stream_and_persist(
                        chat, request, persist_info
                    )
                else:
                    result = self._grok_stream(chat, request)
            else:
                result = await self._grok_generate(chat, request)

                # Persist response (non-streaming only)
            if persist_info and isinstance(result, dict):
                await self._persist_response(
                    persist_info["request_id"],
                    persist_info["request_tstamp"],
                    result,
                    request,
                )

            self.logger.info(
                "Grok generation completed",
                extra={"obj": {"save_mode": request.save_mode}},
            )
            return result

        except Exception as exc:                                                          # Catch SDK / network / any unexpected error
            if isinstance(exc, (ConnectionError, TimeoutError)):
                self.logger.error(
                    "Grok API connection failure", extra={"obj": {"error": str(exc)}}
                )
                raise GrokAPIConnectionError("Failed to connect to Grok API") from exc
            if (
                "unauthenticated" in str(exc).lower()
                or "invalid api key" in str(exc).lower()
            ):
                self.logger.error(
                    "Grok API authentication failure",
                    extra={"obj": {"error": str(exc)}},
                )
                raise GrokAPIAuthenticationError(
                    "Invalid or missing XAI_API_KEY"
                ) from exc
            if "rate limit" in str(exc).lower():
                self.logger.error(
                    "Grok API rate limit exceeded", extra={"obj": {"error": str(exc)}}
                )
                raise GrokAPIRateLimitError("Rate limit exceeded") from exc

            self.logger.error(
                "Unexpected error during Grok generation",
                extra={"obj": {"error": str(exc), "model": request.model}},
            )
            raise wrap_grok_api_error(exc, "Grok API call failed") from exc

    async def _grok_generate(self, chat: Any, request: GrokRequest) -> dict[str, Any]:
        """Non-streaming helper (wrapped for logging)."""
        try:
            sdk_response = await chat.sample()
            return {
                "output": getattr(sdk_response, "content", str(sdk_response)),
                "model": request.model,
                "finish_reason": getattr(sdk_response, "finish_reason", None),
                "raw": getattr(
                    sdk_response,
                    "proto",
                    sdk_response.__dict__ if hasattr(sdk_response, "__dict__") else {},
                ),
            }
        except Exception as exc:
            self.logger.error(
                "Non-streaming generation failed", extra={"obj": {"error": str(exc)}}
            )
            raise wrap_grok_api_error(exc, "Non-streaming generation failed") from exc

    async def _grok_stream(
        self, chat: Any, request: GrokRequest
    ) -> AsyncIterator[LLMStreamingChunkProtocol]:
        """Streaming helper (wrapped for logging)."""
        try:
            async for _full, chunk in chat.stream():
                yield GrokStreamingChunk(
                    text=getattr(chunk, "content", ""),
                    finish_reason=getattr(chunk, "finish_reason", None),
                    is_final=getattr(chunk, "is_final", False),
                    raw={"chunk": chunk},
                )
        except Exception as exc:
            self.logger.error(
                "Streaming generation failed", extra={"obj": {"error": str(exc)}}
            )
            raise wrap_grok_api_error(exc, "Streaming generation failed") from exc

        # ------------------------------------------------------------------
        # Batch processing (with Pyrefly type-ignore annotations)
        # ------------------------------------------------------------------

    async def create_batch(self, batch_name: str) -> dict[str, Any]:
        try:
            batch = await self._client.batch.create(batch_name=batch_name)                # type: ignore[attr-defined]
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
            raise GrokClientBatchError("Failed to create batch") from exc

    async def add_to_batch(self, batch_id: str, requests: list[GrokRequest]) -> None:
        if not batch_id:
            self.logger.error("add_to_batch called with empty batch_id")
            raise GrokClientBatchError("batch_id is required") from None
        try:
            batch_requests = []
            for req in requests:
                # chat = self._client.chat.create(
                #     model=req.model,
                #     batch_request_id=req.batch_request_id,                                # type: ignore[call-arg]
                # )
                chat = self._client.chat.create(
                    model=req.model,
                    batch_request_id=req.batch_request_id,                                # type: ignore[call-arg]
                    store_messages=True,                                                  # Enables modern Responses API
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
                                    parts.append(file(part["file_url"]))                  # placeholder
                                else:
                                    self.logger.error(
                                        "Unsupported multimodal content type",
                                        extra={"obj": {"part": part}},
                                    )
                                    raise GrokClientMultimodalError(
                                        "Unsupported content type in multimodal message"
                                    ) from None
                            chat.append(user(*parts))
                    else:
                        chat.append(user(content))

                batch_requests.append(chat)

                # capture the batch.
            await self._persist_batch_requests(batch_id, requests)

            await self._client.batch.add(                                                 # type: ignore[attr-defined]
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
            raise GrokClientBatchError("Failed to add requests to batch") from exc

    async def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        try:
            batch = await self._client.batch.get(batch_id=batch_id)                       # type: ignore[attr-defined]
            return {
                "batch_id": batch_id,
                "state": getattr(batch, "state", None),
            }
        except Exception as exc:
            self.logger.error(
                "Failed to get batch status",
                extra={"obj": {"batch_id": batch_id, "error": str(exc)}},
            )
            raise GrokClientBatchError("Failed to get batch status") from exc

    async def retrieve_batch_results(
        self, batch_id: str, limit: int = 100
    ) -> dict[str, Any]:
        try:
            results = await self._client.batch.list_batch_results(                        # type: ignore[attr-defined]
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
            raise GrokClientBatchError("Failed to retrieve batch results") from exc

    async def retrieve_and_persist_batch_results(
        self, batch_id: str, limit: int = 100
    ) -> dict[str, Any]:
        """Retrieve batch results from xAI SDK and automatically persist all succeeded responses.
        All responses will carry the same batch_id in meta for easy identification/grouping."""
        results = await self.retrieve_batch_results(batch_id, limit)

        if results.get("succeeded"):
            await self._persist_batch_results(batch_id, results)

        return results

    # ------------------------------------------------------------------
    ## Postgres persistence (infopypg only)
    # ------------------------------------------------------------------

    async def _get_pool(self) -> Any:
        if self._pool is None:
            if self._pg_resolved_settings is None or PgPoolManager is None:
                self.logger.error(
                    "Postgres persistence requested without resolved settings"
                )
                raise GrokInfopypgError(
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

    async def _get_or_create_provider_id(self, name: str = "xai") -> int:
        """Return provider_id for 'xai'; create if missing (normalisation)."""
        pool = await self._get_pool()
        # Try to retrieve
        result = await execute_query(
            pool,
            "SELECT id FROM providers WHERE name = $1",
            params=[name],
            fetch=True,
            logger=self.logger,
        )
        if result and result[0].get("id") is not None:
            return result[0]["id"]

        # Insert on conflict (idempotent)
        await execute_query(
            pool,
            """
            INSERT INTO providers (name, description)
            VALUES ($1, $2)
            ON CONFLICT (name) DO NOTHING
            """,
            params=[name, "xAI"],
            fetch=False,
            logger=self.logger,
        )
        # Re-query
        result = await execute_query(
            pool,
            "SELECT id FROM providers WHERE name = $1",
            params=[name],
            fetch=True,
            logger=self.logger,
        )
        return result[0]["id"] if result else 1                                           # fallback

    def _build_endpoint(self, request: GrokRequest) -> dict[str, Any]:
        """Consistent endpoint metadata for both tables."""
        return {
            "provider": "xai",
            "model": request.model,
            "host": "api.x.ai",
            "endpoint_path": "/v1/chat/completions",
            "prompt_cache_key": request.prompt_cache_key,
        }

    async def _persist_request(
        self,
        request: GrokRequest,
        batch_id: str | None = None,
        batch_index: int | None = None,
    ) -> tuple[uuid.UUID, datetime]:
        """Persist to `requests` table BEFORE API call; returns (request_id, tstamp).
        Now supports optional batch_id and batch_index for reliable batch-result matching."""
        provider_id = await self._get_or_create_provider_id()
        request_id = uuid.uuid4()
        endpoint = self._build_endpoint(request)

        # Full request payload (serialisable via asdict)
        request_payload = asdict(request)

        meta = {
            "conversation_id": self.conversation_id or "unknown",
            "prompt_cache_key": request.prompt_cache_key,
            "batch_request_id": request.batch_request_id,
            "batch_id": batch_id,
            "batch_index": batch_index,
        }

        # Ensure partition exists for today's date (infopypg helper)
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

    async def _generate_stream_and_persist(
        self,
        chat: Any,
        request: GrokRequest,
        persist_info: dict,
    ) -> AsyncIterator[LLMStreamingChunkProtocol]:
        """Yields streaming chunks to the caller AND persists ONE final response row at completion.
        Accumulation avoids per-chunk row explosion while still capturing the complete output."""
        full_text: list[str] = []
        final_finish_reason: str | None = None

        async for chunk in self._grok_stream(chat, request):
            yield chunk
            if chunk.text:
                full_text.append(chunk.text)
            if getattr(chunk, "is_final", False):
                final_finish_reason = getattr(chunk, "finish_reason", None)

                # Build final result compatible with _persist_response
        final_result = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "".join(full_text)}],
                }
            ],
            "model": request.model,
            "finish_reason": final_finish_reason,
            "raw": {"accumulated_text": "".join(full_text)},
        }

        await self._persist_response(
            persist_info["request_id"],
            persist_info["request_tstamp"],
            final_result,
            request,
        )

    async def _persist_response(
        self,
        request_id: uuid.UUID,
        request_tstamp: datetime,
        api_result: dict[str, Any],
        request: GrokRequest | None = None,                                               # now optional for batch results
        batch_id: str | None = None,
    ) -> None:
        """Persist to `responses` table AFTER successful API call (links via composite FK)."""
        provider_id = await self._get_or_create_provider_id()
        response_id = uuid.uuid4()

        # Save media files before writing to db.
        media_files: list[str] = []
        if request is not None:
            media_files = await self._save_media_files(response_id, request)

            # Build endpoint metadata (safe when original request object is unavailable)
        if request is not None:
            endpoint = self._build_endpoint(request)
        else:
            # Minimal fallback for batch results (model is always present in api_result)
            endpoint = {
                "provider": "xai",
                "model": api_result.get("model", "unknown"),
                "host": "api.x.ai",
                "endpoint_path": "/v1/chat/completions",
                "prompt_cache_key": None,
            }

            # Build full response dict compatible with GrokResponse (for reasoning extraction)
        raw_data = api_result.get("raw", api_result)
        if not isinstance(raw_data, dict):
            raw_data = {
                "output": api_result.get("output"),
                "model": api_result.get("model"),
            }

        grok_resp = GrokResponse.from_dict(raw_data)
        response_payload = {
            **raw_data,
            "text": grok_resp.text,
            "tool_calls": grok_resp.tool_calls,
        }

        meta = {
            "conversation_id": self.conversation_id or "unknown",
            "reasoning_text": grok_resp.reasoning_text,
            "finish_reason": api_result.get("finish_reason"),
            "batch_id": batch_id,
            "media_files": media_files,
        }

        # Ensure partition
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

    async def _persist_batch_results(
        self, batch_id: str, sdk_results: dict[str, Any]
    ) -> None:
        """Persist succeeded batch responses by matching on batch_index (stateless, DB-driven)."""
        pool = await self._get_pool()

        # Fetch all requests for this batch, ordered exactly as they were submitted
        requests_data = await execute_query(
            pool,
            """
            SELECT request_id, tstamp 
            FROM requests 
            WHERE meta->>'batch_id' = $1 
            ORDER BY (meta->>'batch_index')::int ASC
            """,
            params=[batch_id],
            fetch=True,
            logger=self.logger,
        )

        # Type-guard: ensure we have data (satisfies Pyrefly)
        if not requests_data:
            self.logger.warning(
                "No requests found for batch – nothing to persist",
                extra={"obj": {"batch_id": batch_id}},
            )
            return

        succeeded = sdk_results.get("succeeded", [])
        persisted_count = 0

        for i, result_item in enumerate(succeeded):
            if i >= len(requests_data):
                self.logger.warning(
                    "More succeeded results than persisted requests for batch",
                    extra={"obj": {"batch_id": batch_id}},
                )
                break

            req_row = requests_data[i]
            request_id = uuid.UUID(req_row["request_id"])
            request_tstamp = req_row["tstamp"]

            # Build API result compatible with _persist_response
            api_result = {
                "raw": result_item
                if isinstance(result_item, dict)
                else vars(result_item)
                if hasattr(result_item, "__dict__")
                else {"result": result_item},
                "model": getattr(result_item, "model", "unknown"),
                "finish_reason": getattr(result_item, "finish_reason", None),
                "output": getattr(result_item, "response", None)
                or getattr(result_item, "content", None),
            }

            await self._persist_response(
                request_id=request_id,
                request_tstamp=request_tstamp,
                api_result=api_result,
                request=None,                                                             # allowed now
                batch_id=batch_id,
            )
            persisted_count += 1

        self.logger.info(
            "Batch responses persisted",
            extra={"obj": {"batch_id": batch_id, "succeeded_count": persisted_count}},
        )

    async def _persist_batch_requests(
        self, batch_id: str, requests: list[GrokRequest]
    ) -> None:
        """Persist every request in a batch with batch_index for later result matching."""
        for i, req in enumerate(requests):
            await self._persist_request(req, batch_id=batch_id, batch_index=i)
            self.logger.info(
                "Batch request persisted",
                extra={"obj": {"batch_id": batch_id, "index": i, "model": req.model}},
            )

    async def _save_media_files(
        self,
        response_id: uuid.UUID,
        request: GrokRequest,
    ) -> list[str]:
        """Save multimodal images/files to the monthly/response_id folder structure.
        Returns list of relative paths (from media root) for storage in responses.meta.
        """
        # Extract prompt snippet (first 100 chars of first user message)
        prompt_snippet = ""
        for msg in request.input.messages:
            if msg.role == "user":
                content = msg.content
                if isinstance(content, str):
                    prompt_snippet = content[:100]
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "input_text":
                            prompt_snippet = part.get("text", "")[:100]
                            break
                break

        media_items: list[dict[str, str]] = []
        for msg in request.input.messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if part.get("type") in ("input_image", "input_file"):
                        url_or_path = part.get("image_url") or part.get("file_url")
                        if url_or_path:
                            media_items.append(
                                {
                                    "type": part["type"],
                                    "url_or_path": url_or_path,
                                    "original_name": Path(url_or_path).name or "file",
                                }
                            )

        if not media_items:
            return []

        # Monthly folder
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
                    # Download from URL
                    def _download():
                        urllib.request.urlretrieve(src, str(dest_path))

                    await asyncio.to_thread(_download)
                else:
                    # Local file – copy
                    await asyncio.to_thread(shutil.copy2, src, dest_path)
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

            def _append_index():
                index_path.parent.mkdir(parents=True, exist_ok=True)
                with index_path.open("a", encoding="utf-8") as f:
                    f.write(index_line)

            await asyncio.to_thread(_append_index)

        return relative_paths
