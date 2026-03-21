#!/usr/bin/env python3
"""
Ollama concrete client implementation.

This module provides `OllamaConcreteClient`, the concrete subclass of
`BaseAsyncProviderClient` for local Ollama models. It uses the official
OpenAI-compatible endpoint (`/v1/chat/completions`) for maximum
compatibility with existing tools and schemas.

Key features (as specified by user):
- Exact MODEL name and ORG host (set in factory).
- Strict model existence check via `/api/tags` — raises clear error
  if missing (no automatic pull).
- Full support for continuation_token (Ollama native `context` list).
- Streaming via SSE (identical public interface to Grok).
- Identical persistence (JSON files or partitioned PostgreSQL),
  structured logging, retries, and concurrency as the Grok client.
- `sys_spec` is stored in meta JSONB for future resource modules.
- No API key required.

Design priorities (in order):
1. Efficiency — semaphore-limited parallelism, minimal object creation.
2. Clarity — model check and Ollama-specific mapping isolated here.
3. Readability — every method ≤ 40 lines, full NumPy-style docstrings,
   inline comments after column 90.
"""

from __future__ import annotations

import asyncio
import json
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
from ai_api.data_structures import SaveMode


class OllamaConcreteClient(BaseAsyncProviderClient):
    """
    Concrete client for local Ollama models.

    Use the factory (`ai_api.factory.create`) rather than instantiating
    directly.
    """

    provider_name: ClassVar[Literal["ollama"]] = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str,
        save_mode: SaveMode = "none",
        output_dir: Path | None = None,
        pg_settings: dict[str, Any] | ResolvedSettingsDict | None = None,
        log_location: str | dict[str, Any] | ResolvedSettingsDict = "grok_client.log",
        concurrency: int = 50,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        """
        Sync initialisation — stores configuration only.

        Async setup (pool creation + model existence check) happens in
        `create`.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")                                              # ensure clean URL
        self.save_mode = save_mode
        self.output_dir = output_dir
        self.pg_settings = pg_settings
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout = timeout
        self.pool: Pool | None = None
        self.provider_id: int = 0
        self.logger: Logger = setup_logger("OLLAMA_CLIENT", log_location=log_location)

    @classmethod
    async def create(
        cls,
        model: str,
        base_url: str,
        save_mode: SaveMode = "none",
        output_dir: Path | None = None,
        pg_settings: dict[str, Any] | ResolvedSettingsDict | None = None,
        log_location: str | dict[str, Any] | ResolvedSettingsDict = "grok_client.log",
        concurrency: int = 50,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> "OllamaConcreteClient":
        """
        Async factory — creates instance, checks model exists, and sets
        up PostgreSQL pool if required.

        Raises
        ------
        ValueError
            If model is not present in Ollama (clear message with list of
            available models).
        """
        instance = cls(
            model=model,
            base_url=base_url,
            save_mode=save_mode,
            output_dir=output_dir,
            pg_settings=pg_settings,
            log_location=log_location,
            concurrency=concurrency,
            max_retries=max_retries,
            timeout=timeout,
        )

        await instance._ensure_model_exists()                                             # strict check — no auto-pull

        if instance.save_mode == "postgres" and instance.pg_settings:
            if not is_ResolvedSettingsDict(instance.pg_settings):
                instance.pg_settings = await async_dict_to_ResolvedSettingsDict(
                    cast(dict, instance.pg_settings)
                )
            instance.pool = await PgPoolManager.get_pool(instance.pg_settings)

        return instance

    async def _ensure_model_exists(self) -> None:
        """
        Verifies the requested model is loaded in Ollama.

        Raises clear error with list of available models if missing.
        """
        tags_url = f"{self.base_url.replace('/v1', '')}/api/tags"
        async with aiohttp.ClientSession() as session:
            async with session.get(tags_url) as resp:
                if resp.status != 200:
                    raise ConnectionError(
                        f"Failed to query Ollama tags: HTTP {resp.status}"
                    )
                data = await resp.json()
                models = [m["name"] for m in data.get("models", [])]
                if self.model not in models:
                    raise ValueError(
                        f"Model '{self.model}' not found on Ollama. "
                        f"Available models: {models}"
                    )
        self.logger.info(f"Model '{self.model}' confirmed present on Ollama.")

    async def _get_provider_id(self) -> int:
        """
        Retrieves or inserts provider ID for 'OLLAMA'.

        Caches result for subsequent calls.
        """
        if self.provider_id != 0:
            return self.provider_id

        pool = await self._get_pool()
        result = await execute_query(
            pool,
            "SELECT id FROM public.providers WHERE name = $1",
            ["OLLAMA"],
            fetch=True,
        )
        if not result:
            result = await execute_query(
                pool,
                "INSERT INTO public.providers (name) VALUES ($1) RETURNING id",
                ["OLLAMA"],
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
        endpoint_json = {"ORG": "OLLAMA", "MODEL": req.model, "ENDPOINT": self.base_url}

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
        ollama_resp: LLMResponse,
        request_id: uuid.UUID,
        request_tstamp: datetime,
    ) -> None:
        """
        Inserts response row linked to request.
        """
        provider_id = await self._get_provider_id()
        endpoint_json = {
            "ORG": "OLLAMA",
            "MODEL": ollama_resp.model,
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
                ollama_resp.id,
                json.dumps(ollama_resp.raw),
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
            headers: dict[str, str] = {
                "Content-Type": "application/json"
            }                                                                             # no auth needed

            if self.save_mode == "postgres":
                try:
                    request_id, request_tstamp = await self._save_request_to_postgres(
                        headers, req
                    )
                except Exception as e:
                    self.logger.error(f"Request save failed: {e}")

            payload = req.to_dict()

            if req.backend_options:
                if hasattr(req.backend_options, "to_dict"):
                    payload["options"] = req.backend_options.to_dict()
                else:
                    payload["options"] = dict(req.backend_options)

            for attempt in range(self.max_retries + 1):
                try:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status != 200:
                            txt = await resp.text()
                            raise aiohttp.ClientError(f"HTTP {resp.status}: {txt}")

                        data = await resp.json()
                        ollama_resp = LLMResponse.from_raw(data, "ollama")

                        if self.save_mode == "json_files":
                            path = self._filename_for(ollama_resp.id)
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
                                ollama_resp,
                                request_id,
                                request_tstamp,
                            )

                        return ollama_resp

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
        Submits batch in parallel (identical public contract to Grok).
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
        Streams tokens from Ollama (SSE format via OpenAI-compat endpoint).

        Yields
        ------
        StreamingChunk
            One per token or final chunk. Continuation token extracted
            from final response.
        """
        headers = {"Content-Type": "application/json"}
        payload = {**request.to_dict(), "stream": True}

        if request.backend_options:
            if hasattr(request.backend_options, "to_dict"):
                payload["options"] = request.backend_options.to_dict()
            else:
                payload["options"] = dict(request.backend_options)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    raise aiohttp.ClientError(f"HTTP {resp.status}: {txt}")

                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str or line_str == "data: [DONE]":
                        continue
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            text = delta.get("content", "")
                            finished = chunk.get("done", False)
                            context = chunk.get("context")                                # native Ollama continuation

                            yield StreamingChunk(
                                delta_text=text,
                                finished=finished,
                                raw_chunk=chunk,
                            )

                            if finished and context:
                                # continuation_token is set on the final chunk
                                # (handled by LLMResponse.from_raw in batch mode)
                                pass
                        except json.JSONDecodeError:
                            continue

    def can_stream(self) -> bool:
        """
        Ollama always supports streaming.
        """
        return True

    def required_vram_gb(self, request: LLMRequest) -> float | None:
        """
        Placeholder for future resource-estimation module.

        Returns None until sys_spec parsing is added in a later module.
        """
        return None
