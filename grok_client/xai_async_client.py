#!/usr/bin/env python3
"""
xai_async_client – generic, robust async client for the xAI Grok API.

Author: Simon Watson
License: MIT

Features
--------
- Fully asynchronous parallel requests with configurable concurrency
- Automatic retries with exponential backoff
- Structured response saving:
      • local JSON files
      • PostgreSQL table 'responses' (with JSONB columns)
- Optional unique filename generation using user_id, timestamp, response id
- Rich logging and comprehensive error handling
- Type-hinted, extensively documented, Python 3.11+ ready

To install:
-----------
1)  activate the conda environment you wish to install it in. 
2)  From the directory that contains your xai_async_client.py file:
pip install -e .

subsequently, import as here:
from xai_async_client import XAIAsyncClient, GrokRequest

"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, overload

import aiohttp
import asyncpg  # type: ignore[import-untyped]

# -------------------------------------------------------------------------------------- #
# Configuration & Types
# -------------------------------------------------------------------------------------- #

SaveMode = Literal["none", "json_files", "postgres"]

class GrokMessage(TypedDict):
    """Message format expected by the xAI /chat/completions endpoint."""
    role: Literal["system", "user", "assistant"]
    content: str


class ResolvedSettingsDict(TypedDict):
    """
    Typed dictionary for resolved PostgreSQL settings after validation.

    All values are guaranteed str except tablespace_path and extensions.
    """
    db_user: str
    db_host: str
    db_port: str
    db_name: str
    tablespace_name: str
    tablespace_path: str | None
    password: str
    extensions: list[str] | None


@dataclass(frozen=True, slots=True)
class GrokRequest:
    """Immutable container for a single API request."""
    messages: list[GrokMessage]
    model: str = "grok-beta"
    temperature: float = 0.9
    max_tokens: int | None = None
    user_id: str | None = None                                                            # custom identifier → payload "user"
    metadata: dict[str, Any] | None = None                                                # extra tags for future use


@dataclass(slots=True)
class GrokResponse:
    """Complete response object returned to caller."""
    request: GrokRequest
    content: str
    response_id: str
    finish_reason: str
    usage: dict[str, int]
    raw: dict[str, Any]                                                                   # full JSON from xAI


# -------------------------------------------------------------------------------------- #
# Core Client
# -------------------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)


class XAIAsyncClient:
    """
    High-level async client for batched Grok API calls with robust saving options.

    Supports saving to local JSON files **or** directly into a PostgreSQL table
    named `` Loot.responses`` with the schema defined below.

    PostgreSQL Table Schema
    -----------------------
    .. code-block:: sql

        CREATE TABLE IF NOT EXISTS responses (
            idx        BIGSERIAL PRIMARY KEY,
            received   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            endpoint   JSONB NOT NULL,
            response   JSONB NOT NULL,
            note       TEXT
        );

    Example
    -------
    >>> from xai_async_client import XAIAsyncClient, GrokRequest
    >>>
    >>> pg_settings: ResolvedSettingsDict = {
    ...     "db_user": "simon", "db_host": "localhost", "db_port": "5432",
    ...     "db_name": "grok_logs", "tablespace_name": "pg_default",
    ...     "tablespace_path": None, "password": "secret",
    ...     "extensions": ["pgcrypto"]
    ... }
    >>>
    >>> client = XAIAsyncClient(
    ...     api_key="xai_...", save_mode="postgres", pg_settings=pg_settings
    ... )
    >>> req = GrokRequest(messages=[{"role": "user", "content": "Hello"}],
    ...                   user_id="test_001")
    >>> results = await client.submit_batch([req])
    """

    def __init__(
        self,
        api_key: str,
        *,
        save_mode: SaveMode = "none",
        output_dir: Path | None = None,
        pg_settings: ResolvedSettingsDict | None = None,
        concurrency: int = 50,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialise the client.

        Parameters
        ----------
        api_key
            xAI API key (or set XAI_API_KEY environment variable).
        save_mode
            ``"none"`` → only return objects  
            ``"json_files"`` → write each response to disk  
            ``"postgres"`` → insert into PostgreSQL table ``responses``
        output_dir
            Required when ``save_mode="json_files"``.
        pg_settings
            Required when ``save_mode="postgres"``. Must match ``ResolvedSettingsDict``.
        concurrency
            Maximum simultaneous HTTP requests.
        timeout
            Per-request timeout in seconds.
        max_retries
            Number of retries on transient failures.
        """
        self.api_key = api_key
        self.base_url = "https://api.x.ai/v1/chat/completions"
        self.save_mode = save_mode
        self.output_dir = output_dir
        self.pg_settings = pg_settings
        self.concurrency = concurrency
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries

        # Validation
        if save_mode == "json_files":
            if not output_dir or not output_dir.is_dir():
                raise ValueError(
                    "output_dir must be an existing directory for json_files mode")
        elif save_mode == "postgres":
            if not pg_settings:
                raise ValueError("pg_settings is required when save_mode='postgres'")

        self._pg_pool: asyncpg.Pool | None = None

    async def _ensure_pg_pool(self) -> asyncpg.Pool:
        """Lazily create and return a connection pool (singleton per client)."""
        if self._pg_pool is None:
            if not self.pg_settings:
                raise RuntimeError("PostgreSQL settings not configured")
            dsn = (
                f"postgres://{self.pg_settings['db_user']}:{
                self.pg_settings['password']}"
                f"@{self.pg_settings['db_host']}:{self.pg_settings['db_port']}/{
                self.pg_settings['db_name']}"
            )
            self._pg_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=10)
        return self._pg_pool

    # ---------------------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------------------- #

    def _make_payload(self, req: GrokRequest) -> dict[str, Any]:
        """Convert GrokRequest → JSON payload for the API."""
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": req.messages,
            "temperature": req.temperature,
        }
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.user_id is not None:
            payload["user"] = req.user_id
        if req.metadata:
            payload["metadata"] = req.metadata
        return payload

    def _filename_for(self, req: GrokRequest, resp_id: str) -> Path:
        """Generate deterministic filename for JSON file saving."""
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe_uid = (req.user_id or "no_user_id").replace("/", "_")
        return self.output_dir / f"{ts}_{safe_uid}_{resp_id}.json"                        # type: ignore[arg-type]

    async def _save_to_postgres(self, resp: GrokResponse) -> None:
        """
        Insert a single response into the ``responses`` table.

        The ``endpoint`` JSONB column is populated with::
            {
                "ORG": "GROK",
                "MODEL": <model name>,
                "ENDPOINT": "https://api.x.ai/v1/chat/completions"
            }
        """
        pool = await self._ensure_pg_pool()
        endpoint_json = {
            "ORG": "GROK",
            "MODEL": resp.request.model,
            "ENDPOINT": self.base_url,
        }
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO responses (endpoint, response, note)
                VALUES ($1::jsonb, $2::jsonb, $3)
                """,
                endpoint_json,
                resp.raw,
                resp.request.user_id or None,
            )
        logger.info(f"Inserted response {resp.response_id} into PostgreSQL")

    async def _single_call(
        self,
        session: aiohttp.ClientSession,
        req: GrokRequest,
        semaphore: asyncio.Semaphore,
    ) -> GrokResponse | Exception:
        """
        Perform one API call with retries and optional persistence.

        Returns a ``GrokResponse`` on success or an ``Exception`` on final failure.
        """
        async with semaphore:
            payload = self._make_payload(req)
            headers = {"Authorization": f"Bearer {self.api_key}"}

            for attempt in range(self.max_retries + 1):
                try:
                    async with session.post(
                        self.base_url, json=payload, headers=headers,
                        timeout=self.timeout
                    ) as resp:
                        if resp.status != 200:
                            txt = await resp.text()
                            raise aiohttp.ClientError(f"HTTP {resp.status}: {txt}")

                        data = await resp.json()
                        choice = data["choices"][0]
                        grok_resp = GrokResponse(
                            request=req,
                            content=choice["message"]["content"],
                            response_id=data["id"],
                            finish_reason=choice["finish_reason"],
                            usage=data.get("usage", {}),
                            raw=data,
                        )

                        # Persistence
                        if self.save_mode == "json_files":
                            path = self._filename_for(req, data["id"])
                            path.write_text(json.dumps(
                                data, ensure_ascii=False, indent=2))
                            logger.info(f"Saved response → {path.name}")

                        elif self.save_mode == "postgres":
                            await self._save_to_postgres(grok_resp)

                        return grok_resp

                except (aiohttp.ClientError, asyncio.TimeoutError,
                        json.JSONDecodeError, asyncpg.PostgresError) as e:
                    if attempt == self.max_retries:
                        logger.error(f"Permanent failure for {
                        req.user_id or 'unknown'}: {e}")
                        return e
                    wait = 1.5 ** attempt
                    logger.warning(f"Retry {attempt+1}/{self.max_retries} for {
                    req.user_id}: {e}")
                    await asyncio.sleep(wait)

            return RuntimeError("Unreachable")

    # ---------------------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------------------- #

    @overload
    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: Literal[True]
    ) -> list[GrokResponse]: ...

    @overload
    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: Literal[False]
    ) -> None: ...

    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: bool = True
    ) -> list[GrokResponse] | None:
        """
        Submit many requests in parallel with full concurrency control.

        Parameters
        ----------
        requests
            List of ``GrokRequest`` objects to send.
        return_responses
            If ``False``, only saves (to disk or DB) and returns ``None``.
            Useful for fire-and-forget logging jobs.

        Returns
        -------
        List[GrokResponse] | None
            Successful responses (failed tasks are logged but omitted).
        """
        if not requests:
            return [] if return_responses else None

        semaphore = asyncio.Semaphore(self.concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._single_call(session, req, semaphore) for req in requests
            ]
            results = await asyncio.gather(
                *tasks, return_exceptions=False)

        responses: list[GrokResponse] = []
        for item in results:
            if isinstance(item, Exception):
                logger.error(f"Task failed permanently: {item}")
            else:
                responses.append(item)

        return responses if return_responses else None

    async def close(self) -> None:
        """Close the PostgreSQL connection pool if open (call on shutdown)."""
        if self._pg_pool is not None:
            await self._pg_pool.close()
#  LocalWords:  Authorization untyped
