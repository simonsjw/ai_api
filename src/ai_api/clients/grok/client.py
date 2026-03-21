#!/usr/bin/env python3
"""
Grok Concrete Client for xAI API (Unified Architecture).

This module provides the complete concrete implementation for the xAI Grok API.
It is a direct, feature-complete evolution of the original XAIAsyncClient from
grok_client/grok_client.py.

Every piece of functionality has been preserved:
- Batched async requests with semaphore-limited concurrency
- Exponential-backoff retries on transient errors
- Conversation ID caching (`set_conv_id`) for Grok prompt caching and cost reduction
- Dual persistence modes: "none", "json_files", or "postgres" (with daily partitioned tables)
- Full request + response saving to PostgreSQL (including provider_id lookup and partitioning)
- Structured JSON output support
- Comprehensive error handling and custom logging
- Token-by-token streaming (SSE)

The old `XAIAsyncClient` class name and GrokRequest/GrokResponse types have been adapted
to the new unified types (`LLMRequest`, `LLMResponse`, `StreamingChunk`), but the
behaviour and database schema remain 100% compatible.

Use via the factory:
    client = await create(provider="grok", model=..., api_key=..., save_mode=...)
"""

import asyncio
import json
import traceback
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

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
from ai_api.data_structures import SaveMode


class GrokConcreteClient(BaseAsyncProviderClient):
    """
    Concrete asynchronous client for the xAI Grok API.

    All original legacy behaviour is preserved. Use the factory function instead of
    direct instantiation.
    """

    provider_name: ClassVar[Literal["grok"]] = "grok"
    base_url: str = "https://api.x.ai/v1/responses"
    timeout: float = 60.0

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
        Synchronous initialisation. Async setup (DB pool, etc.) is performed in create().
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

        # Sanitised copy for meta logging (never stores password)
        clean_pg = {
            k: v for k, v in (pg_settings or {}).items() if str(k).upper() != "PASSWORD"
        }
        self._initial_args = {
            "save_mode": save_mode,
            "output_dir": str(output_dir) if output_dir else None,
            "concurrency": concurrency,
            "max_retries": max_retries,
            "timeout": timeout,
            "set_conv_id": set_conv_id,
            "pg_settings": clean_pg,
        }

    @classmethod
    async def create(
        cls,
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
    ) -> "GrokConcreteClient":
        """
        Async factory (recommended). Performs PostgreSQL pool setup when save_mode="postgres".
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
                    cast(dict[str, Any], instance.pg_settings)
                )
            instance.pool = await PgPoolManager.get_pool(instance.pg_settings)

        return instance

    @property
    def conv_id(self) -> str | None:
        """
        Lazily resolved conversation ID for Grok caching (exactly as in original).

        If set_conv_id=True, a UUID is generated once and reused for all requests.
        """
        if isinstance(self._set_conv_id, str):
            return self._set_conv_id
        if self._set_conv_id:
            self._set_conv_id = str(uuid.uuid4())
            return self._set_conv_id
        return None

    async def _get_provider_id(self) -> int:
        """Retrieves or inserts provider ID for 'GROK' (original logic preserved)."""
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
            self.logger.info(msg="Inserted 'GROK' into providers table.")

        if not result:
            raise ValueError("provider_id was not found and could not be created")

        self.provider_id = result[0]["id"]

        return self.provider_id

    def _filename_for(self, resp_id: str) -> Path:
        """Timestamped filename for JSON persistence (original behaviour)."""
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return (self.output_dir or Path(".")) / f"{ts}_{resp_id}.json"

    async def _get_pool(self) -> Pool:
        """Returns PostgreSQL pool (original logic with safety checks)."""
        if self.save_mode != "postgres" or self.pg_settings is None:
            raise ValueError("PostgreSQL settings required for this mode.")
        if not is_ResolvedSettingsDict(self.pg_settings):
            self.pg_settings = await async_dict_to_ResolvedSettingsDict(
                cast(dict[str, Any], self.pg_settings)
            )
        return await PgPoolManager.get_pool(self.pg_settings)

    # === Full Postgres persistence methods (identical to original) ===
    async def _save_request_to_postgres(
        self, headers: dict[str, Any], req: LLMRequest
    ) -> tuple[uuid.UUID, datetime]:
        """Inserts request into partitioned `requests` table (original logic)."""
        request_id = uuid.uuid4()
        provider_id = await self._get_provider_id()
        endpoint_json = {"ORG": "GROK", "MODEL": req.model, "ENDPOINT": self.base_url}

        pool = await self._get_pool()
        tstamp = datetime.now(UTC)

        await ensure_partition_exists(
            connection_pool=pool,
            table_name="requests",
            target_date=None,
            range_interval="daily",
            look_ahead_days=2,
        )

        await execute_query(
            pool,
            """
            INSERT INTO public.requests (tstamp, provider_id, endpoint, request_id, request, meta)
            VALUES ($1, $2, $3::jsonb, $4, $5::jsonb, $6::jsonb) RETURNING tstamp
            """,
            [
                tstamp,
                provider_id,
                json.dumps(endpoint_json),
                request_id,
                json.dumps(req.to_dict()),
                json.dumps({**self._initial_args, "headers": headers}),
            ],
            fetch=True,
        )
        return request_id, tstamp

    async def _save_response_to_postgres(
        self,
        meta: dict[str, Any],
        resp: LLMResponse,
        request_id: uuid.UUID,
        request_tstamp: datetime,
    ) -> None:
        """Inserts response linked to request (original logic)."""
        provider_id = await self._get_provider_id()
        endpoint_json = {"ORG": "GROK", "MODEL": resp.model, "ENDPOINT": self.base_url}

        pool = await self._get_pool()
        tstamp = datetime.now(UTC)

        await ensure_partition_exists(
            connection_pool=pool,
            table_name="responses",
            target_date=None,
            range_interval="daily",
            look_ahead_days=2,
        )

        await execute_query(
            pool,
            """
            INSERT INTO public.responses (tstamp, provider_id, endpoint, request_id, request_tstamp,
                                          response_id, response, meta)
            VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7::jsonb, $8::jsonb)
            """,
            [
                tstamp,
                provider_id,
                json.dumps(endpoint_json),
                request_id,
                request_tstamp,
                resp.id,
                json.dumps(resp.raw),
                json.dumps(meta),
            ],
            fetch=False,
        )

        # === Core request methods (original + unified types) ===

    async def _single_call(
        self,
        session: aiohttp.ClientSession,
        req: LLMRequest,
        semaphore: asyncio.Semaphore,
    ) -> LLMResponse | Exception:
        """Single request with retries, persistence, and conv_id (original core)."""
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
                        response = LLMResponse.from_raw(data, provider="grok")

                        if self.save_mode == "json_files" and self.output_dir:
                            path = self._filename_for(response.id)
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
                                response,
                                request_id,
                                request_tstamp,
                            )

                        return response

                except Exception as e:
                    if attempt == self.max_retries:
                        self.logger.error(f"Permanent failure: {e}")
                        return e
                    await asyncio.sleep(1.5**attempt)

            return RuntimeError("Unreachable")

    async def submit_batch(
        self, requests: list[LLMRequest], return_responses: bool = True
    ) -> list[LLMResponse] | None:
        """Batch submission (original overload behaviour preserved)."""
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

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        """Token-by-token streaming (new but fully compatible with original persistence)."""
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
                    raise aiohttp.ClientError(f"Stream HTTP {resp.status}: {txt}")

                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                            text = delta.get("content", "")
                            yield StreamingChunk(
                                delta_text=text,
                                finished=chunk_data.get("done", False),
                                raw_chunk=chunk_data,
                            )
                        except json.JSONDecodeError:
                            continue

    def can_stream(self) -> bool:
        return True

    def required_vram_gb(self, request: LLMRequest) -> float | None:
        return None
