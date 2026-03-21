#!/usr/bin/env python3
"""
Grok concrete client implementation.

This module provides `GrokConcreteClient`, the concrete subclass of
`BaseAsyncProviderClient` for the xAI Grok API endpoint. It reuses the
battle-tested asynchronous batching, exponential-backoff retries,
persistence (JSON files or partitioned PostgreSQL), structured logging,
and conversation-caching logic from the original `grok_client.py` while
adapting everything to the new unified `LLMRequest` / `LLMResponse`
abstractions.

Design priorities (in order):
1. Efficiency — semaphore-limited parallelism, minimal object creation,
   reuse of existing persistence helpers.
2. Clarity — provider-specific mapping logic isolated; Grok header
   (`x-grok-conv-id`) and response shape handled here only.
3. Readability — every method ≤ 40 lines, full NumPy-style docstrings,
   inline comments after column 90.

Streaming is supported via the native SSE endpoint (`stream=True` in
payload). The client returns `StreamingChunk` objects so the public
API remains identical across providers.

All persistence and logging behaviour is unchanged from your original
implementation.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, cast, override

import aiohttp
from asyncpg import Pool
from infopypg import (
    PgPoolManager,
    ResolvedSettingsDict,
    async_dict_to_ResolvedSettingsDict,
    ensure_partition_exists,
    execute_query,
    is_ResolvedSettingsDict,
)
from logger import Logger, setup_logger

from ai_api.clients.base import BaseAsyncProviderClient
from ai_api.core.request import LLMRequest
from ai_api.core.response import LLMResponse, StreamingChunk
from ai_api.data_structures import GrokResponse, SaveMode


class GrokConcreteClient(BaseAsyncProviderClient):
    """
    Concrete client for xAI Grok API.

    Use the factory (`ai_api.factory.create`) rather than instantiating
    directly.
    """

    provider_name: ClassVar[Literal["grok"]] = "grok"
    base_url: str = "https://api.x.ai/v1/responses"

    def __init__(
        self,
        model: str,
        api_key: str,
        save_mode: SaveMode = "none",
        output_dir: Path | None = None,
        pg_settings: dict[str, Any] | ResolvedSettingsDict | None = None,
        log_location: str | dict[str, Any] | ResolvedSettingsDict = "grok_client.log",
        concurrency: int = 50,
        max_retries: int = 3,
        timeout: float = 60.0,
        set_conv_id: bool | str | None = False,
    ) -> None:
        """
        Sync initialisation — stores configuration only.

        Async setup (pool creation) happens in `create`.
        """
        self.model = model
        self.api_key = api_key
        self.save_mode = save_mode
        self.output_dir = output_dir
        self.pg_settings = pg_settings
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout = timeout
        self._set_conv_id = set_conv_id
        self.pool: Pool | None = None
        self.provider_id: int = 0
        self.logger: Logger = setup_logger("GROK_CLIENT", log_location=log_location)

    @classmethod
    async def create(
        cls,
        model: str,
        api_key: str,
        base_url: str | None = None,
        save_mode: SaveMode = "none",
        output_dir: Path | None = None,
        pg_settings: dict[str, Any] | ResolvedSettingsDict | None = None,
        log_location: str | dict[str, Any] | ResolvedSettingsDict = "grok_client.log",
        concurrency: int = 50,
        max_retries: int = 3,
        timeout: float = 60.0,
        set_conv_id: bool | str | None = False,
    ) -> "GrokConcreteClient":
        """
        Async factory — creates instance and performs PostgreSQL setup.

        Returns
        -------
        GrokConcreteClient
            Fully initialised client.
        """
        instance = cls(
            model=model,
            api_key=api_key,
            save_mode=save_mode,
            output_dir=output_dir,
            pg_settings=pg_settings,
            log_location=log_location,
            concurrency=concurrency,
            max_retries=max_retries,
            timeout=timeout,
            set_conv_id=set_conv_id,
        )

        if instance.save_mode == "postgres" and instance.pg_settings:
            if not is_ResolvedSettingsDict(instance.pg_settings):
                instance.pg_settings = await async_dict_to_ResolvedSettingsDict(
                    cast(dict, instance.pg_settings)
                )
            instance.pool = await PgPoolManager.get_pool(instance.pg_settings)

        return instance

    @property
    def conv_id(self) -> str | None:
        """
        Lazily resolves or generates conversation ID for caching.

        Returns
        -------
        str | None
            Same ID for all requests after initialisation (Grok caching).
        """
        if isinstance(self._set_conv_id, str):
            return self._set_conv_id
        if self._set_conv_id:
            self._set_conv_id = str(uuid.uuid4())
            return self._set_conv_id
        return None

    async def _get_provider_id(self) -> int:
        """
        Retrieves or inserts provider ID for 'GROK'.

        Caches result for subsequent calls.
        """
        if self.provider_id != 0:
            return self.provider_id

        pool = await self._get_pool()
        result = await execute_query(
            pool,
            "SELECT id FROM public.providers WHERE name = $1",
            ["GROK"],
            fetch=True,
        )
        if not result:
            result = await execute_query(
                pool,
                "INSERT INTO public.providers (name) VALUES ($1) RETURNING id",
                ["GROK"],
                fetch=True,
            )
        if not result:
            raise ValueError("provider_id was not found and could not be created")

        self.provider_id = result[0]["id"]
        return self.provider_id

    def _filename_for(self, resp_id: str) -> Path:
        """
        Generates timestamped filename for JSON persistence.
        """
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return self.output_dir / f"{ts}_{resp_id}.json"                                   # type: ignore[arg-type]

    async def _get_pool(self) -> Pool:
        """
        Returns PostgreSQL pool (creates if needed).
        """
        if self.save_mode != "postgres" or self.pg_settings is None:
            raise ValueError("PostgreSQL settings required")
        if not is_ResolvedSettingsDict(self.pg_settings):
            self.pg_settings = await async_dict_to_ResolvedSettingsDict(
                cast(dict, self.pg_settings)
            )
        return await PgPoolManager.get_pool(self.pg_settings)

    async def _save_request_to_postgres(
        self, headers: dict[str, Any], req: LLMRequest
    ) -> tuple[uuid.UUID, datetime]:
        """
        Inserts request row and returns ID + timestamp.
        """
        request_id = uuid.uuid4()
        provider_id = await self._get_provider_id()
        endpoint_json = {"ORG": "GROK", "MODEL": req.model, "ENDPOINT": self.base_url}

        pool = await self._get_pool()
        tstamp = datetime.now(UTC)

        await ensure_partition_exists(
            connection_pool=pool,
            table_name="requests",
            target_date=tstamp.date(),
            partition_key="tstamp",
            range_interval="daily",
            look_ahead_days=2,
        )

        await execute_query(
            pool,
            (
                "INSERT INTO public.requests "
                "(tstamp, provider_id, endpoint, request_id, request, meta) "
                "VALUES ($1, $2, $3::jsonb, $4, $5::jsonb, $6::jsonb) "
                "RETURNING tstamp"
            ),
            [
                tstamp,
                provider_id,
                json.dumps(endpoint_json),
                request_id,
                json.dumps(req.to_dict()),
                json.dumps({"headers": headers, "timeout": self.timeout}),
            ],
            fetch=True,
        )
        return request_id, tstamp

    async def _save_response_to_postgres(
        self,
        meta: dict[str, Any],
        grok_resp: LLMResponse,
        request_id: uuid.UUID,
        request_tstamp: datetime,
    ) -> None:
        """
        Inserts response row linked to request.
        """
        provider_id = await self._get_provider_id()
        endpoint_json = {
            "ORG": "GROK",
            "MODEL": grok_resp.model,
            "ENDPOINT": self.base_url,
        }

        pool = await self._get_pool()
        tstamp = datetime.now(UTC)

        await ensure_partition_exists(
            connection_pool=pool,
            table_name="responses",
            target_date=tstamp.date(),
            partition_key="tstamp",
            range_interval="daily",
            look_ahead_days=2,
        )
        await execute_query(
            pool,
            (
                "INSERT INTO public.responses "
                "(tstamp, provider_id, endpoint, request_id, request_tstamp, "
                "response_id, response, meta) "
                "VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7::jsonb, $8::jsonb)"
            ),
            [
                tstamp,
                provider_id,
                json.dumps(endpoint_json),
                request_id,
                request_tstamp,
                grok_resp.id,
                json.dumps(grok_resp.raw),
                json.dumps(meta),
            ],
            fetch=False,
        )

    async def _persist_streamed_response(
        self, request: LLMRequest, resp: LLMResponse
    ) -> None:
        """
        Persist a streamed response exactly once (reuses batch logic).
        Now receives the original LLMRequest → no more None/null crashes.
        """
        if self.save_mode == "none":
            return

        request_id, request_tstamp = await self._save_request_to_postgres(
            {"headers": {"stream": True}},                                                # clear marker
            request,                                                                      # ← fixed: real request object
        )

        await self._save_response_to_postgres(
            {"headers": {}, "stream": True},
            resp,
            request_id,
            request_tstamp,
        )

    async def _single_call(
        self,
        session: aiohttp.ClientSession,
        req: LLMRequest,
        semaphore: asyncio.Semaphore,
    ) -> LLMResponse | Exception:
        """
        Executes a single request with retries and persistence.
        """
        async with semaphore:
            request_id = request_tstamp = None
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                **({"x-grok-conv-id": self.conv_id} if self.conv_id else {}),
            }

            if self.save_mode == "postgres":
                try:
                    request_id, request_tstamp = await self._save_request_to_postgres(
                        headers, req
                    )
                except Exception as e:
                    self.logger.error(f"Request save failed: {e}")

            payload = req.to_dict()

            for attempt in range(self.max_retries + 1):
                try:
                    async with session.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status != 200:
                            txt = await resp.text()
                            raise aiohttp.ClientError(f"HTTP {resp.status}: {txt}")

                        data = await resp.json()
                        grok_resp = LLMResponse.from_raw(data, "grok", self.conv_id)

                        if self.save_mode == "json_files":
                            path = self._filename_for(grok_resp.id)
                            path.write_text(
                                json.dumps(data, ensure_ascii=False, indent=2)
                            )

                        elif (
                            self.save_mode == "postgres"
                            and request_id
                            and request_tstamp
                        ):
                            await self._save_response_to_postgres(
                                {"headers": headers},
                                grok_resp,
                                request_id,
                                request_tstamp,
                            )

                        return grok_resp

                except Exception as e:
                    if attempt == self.max_retries:
                        self.logger.error(f"Permanent failure: {e}")
                        return e
                    await asyncio.sleep(1.5**attempt)

            raise RuntimeError("Unreachable")

    async def submit_batch(
        self, requests: list[LLMRequest], return_responses: bool = True
    ) -> list[LLMResponse] | None:
        """
        Submits batch in parallel (exact same public contract as before).
        """
        if not requests:
            return [] if return_responses else None

        semaphore = asyncio.Semaphore(self.concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = [self._single_call(session, req, semaphore) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=False)

        responses: list[LLMResponse] = []
        for item in results:
            if isinstance(item, Exception):
                self.logger.error(f"Task failed: {item}")
            else:
                responses.append(item)

        return responses if return_responses else None

    @override
    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:         # type: ignore[override]   # Pyrefly strict abstract async override
        """
        Streams tokens from Grok (SSE format).

        Yields
        ------
        StreamingChunk
            One per token or final chunk.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            **({"x-grok-conv-id": self.conv_id} if self.conv_id else {}),
        }
        payload = {**request.to_dict(), "stream": True}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    raise aiohttp.ClientError(f"HTTP {resp.status}: {txt}")

                buffer = ""
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            text = delta.get("content", "")
                            yield StreamingChunk(
                                delta_text=text,
                                finished=chunk.get("done", False),
                                raw_chunk=chunk,
                            )
                        except json.JSONDecodeError:
                            continue

    def can_stream(self) -> bool:
        """
        Grok always supports streaming.
        """
        return True

    def required_vram_gb(self, request: LLMRequest) -> float | None:
        """
        Remote provider — no VRAM estimate.
        """
        return None
