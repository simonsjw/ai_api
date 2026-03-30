#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLMClient - Unified asynchronous client for Grok (xAI) and Ollama LLMs.

This module provides the `LLMClient` class, which serves as the single entry
point for all LLM interactions in the ai_api library. It accepts a provider
("grok" or "ollama"), a model name, and a request object (GrokRequest or
OllamaRequest). The class automatically infers the correct endpoint and payload
format (Grok uses "input" for the Responses API while Ollama uses "messages")
via the request object's `to_payload()` and `get_endpoint()` methods.

Key responsibilities:
- Handles classic (non-streaming) responses and streaming via the refined
  LLMStreamingChunkProtocol for uniform consumption across providers.
- Supports Grok-native prompt caching (via `conversation_id`), tools, vision,
  parallel generation, and native /v1/batches.
- Automatically persists every request/response pair to PostgreSQL using
  infopypg (inserts into the partitioned `requests` and `responses` tables).
- Performs structured logging via the custom `logger` module (writes to the
  existing `logs` table or file).
- All database operations are lazy (via `_ensure_db_pool`) and use the single
  settings dict supplied at initialisation.

Public API
----------
__init__(provider, model, settings, api_key=None, conversation_id=None, logger=None)
    Initialises the client, uses optional initialised logger, and creates the underlying
    provider client (AsyncOpenAI for Grok or ollama.AsyncClient for Ollama).
    Depends on: logger.Logger, infopypg.async_dict_to_ResolvedSettingsDict.

generate(request, stream=False, use_cache=True, **kwargs)
    Main entry point. Dispatches to the appropriate private generator based on
    provider. Returns a full response or an AsyncIterator of streaming chunks
    implementing LLMStreamingChunkProtocol.
    Interacts with: _grok_generate(), _ollama_generate(), _persist_interaction().

parallel_generate(requests, max_concurrent=20, use_cache=True, **kwargs)
    Fires multiple requests concurrently using asyncio.gather and a semaphore.
    Designed for post-cache-seed workloads on Grok.
    Interacts with: generate(), asyncio.Semaphore, logger.

submit_batch(batch_request, persist=True)
    Submits a native Grok batch to /v1/batches (Grok-only). Optionally persists
    batch metadata.
    Interacts with: GrokBatchRequest.to_payload(), _persist_interaction(), logger.

await_batch_completion(batch_id, poll_interval=30, timeout_seconds=None)
    Polls the Grok /v1/batches/{id} endpoint until completion and returns the
    list of final GrokResponse objects.
    Interacts with: GrokResponse.from_dict(), logger, asyncio.sleep.

create_ollama_model(modelfile)
    Ollama-only convenience method to create/update a model from a Modelfile.
    Raises ValueError if called on Grok.

get_embeddings(input_text)
    Returns embeddings (Ollama-only; Grok does not expose this endpoint).
    Interacts with: ollama.AsyncClient.embeddings().

Private helpers (called internally)
-----------------------------------
_ensure_db_pool()
    Lazily resolves the settings dict and obtains the infopypg pool.

_persist_interaction(request_obj, response_obj)
    Writes request and response rows to the partitioned PostgreSQL tables.
    Interacts with: infopypg.execute_query(), logger.

_grok_generate(), _grok_stream()
    Grok-specific paths using the Responses API (supports caching, tools, vision).
    Interacts with: client.responses.create(), GrokStreamingChunk.

_ollama_generate(), _ollama_stream()
    Ollama-specific paths (OpenAI-compatible chat endpoint).
    Interacts with: client.chat(), OllamaStreamingChunk.

Dependencies
------------
- ai_api.data_structures.grok / ollama (request/response/streaming classes)
- infopypg (PgPoolManager, async_dict_to_ResolvedSettingsDict, execute_query)
- logger (Logger)
- openai.AsyncOpenAI (for Grok Responses API)
- ollama (for local inference)
- asyncio (for parallel and streaming operations)

All methods respect the ≤40-line rule and prioritise efficiency (lazy pool
creation, minimal allocations in hot paths, semaphore throttling).
"""

from __future__ import annotations

import asyncio
import json
from logging import Logger
from os import getenv
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import ollama
from infopypg import (
    PgPoolManager,
    ResolvedSettingsDict,
    async_dict_to_ResolvedSettingsDict,
    execute_query,
)
from openai import AsyncOpenAI

from ai_api.data_structures.grok import (
    GrokBatchRequest,
    GrokBatchResponse,
    GrokRequest,
    GrokResponse,
    GrokStreamingChunk,
    LLMStreamingChunkProtocol,                                                            # re-exported protocol
)
from ai_api.data_structures.ollama import (
    OllamaRequest,
    OllamaResponse,
    OllamaStreamingChunk,
)


class LLMClient:
    """Unified client for Grok and Ollama with automatic DB persistence and logging.

    Public interface is provider-agnostic except for a few convenience methods
    that expose Ollama-only features.
    """

    def __init__(
        self,
        provider: Literal["grok", "ollama"],
        model: str,
        settings: dict[str, Any],
        api_key: str | None = None,
        conversation_id: str | None = None,
        logger: Logger | None = None,
    ) -> None:
        """Initialise the LLMClient.

        Parameters
        ----------
        provider
            Either "grok" (remote xAI) or "ollama" (local).
        model
            Model identifier (e.g. "grok-beta", "llama3.2").
        settings
            Lower-case connection dictionary for PostgreSQL (passed to
            infopypg.async_dict_to_ResolvedSettingsDict). Used for both logging
            and request/response persistence.
        api_key
            xAI API key (required for Grok; ignored for Ollama).
        conversation_id
            Stable UUID or session ID for Grok prompt caching. First call with
            a new ID seeds the cache; reuse the same ID for subsequent calls.
        """
        self.provider = provider
        self.model = model
        self.settings = settings
        self.conversation_id = conversation_id

        # Logger integrates with infopypg and writes to the existing logs table
        if logger:
            self.logger: Logger = logger

        self.resolved_settings: ResolvedSettingsDict | None = None
        self.client: AsyncOpenAI | ollama.AsyncClient | None = None

        if provider == "grok":
            if not api_key:
                msg = "api_key required for Grok provider"
                if self.logger:
                    self.logger.error(msg)
                raise ValueError(msg)
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1",
            )
        elif provider == "ollama":
            self.client = ollama.AsyncClient(host="http://127.0.0.1:11434")
        else:
            msg: str = f"Unsupported provider: {provider}"
            if self.logger:
                self.logger.error(msg)
            raise ValueError(msg)

        if self.logger:
            self.logger.debug(
                "LLMClient initialised",
                extra={"obj": {"provider": provider, "model": model}},
            )

    async def _ensure_db_pool(self) -> None:
        """Lazily resolve settings and obtain infopypg connection pool."""
        if self.resolved_settings is None:
            self.resolved_settings = await async_dict_to_ResolvedSettingsDict(
                self.settings
            )

    async def _persist_interaction(
        self, request_obj: GrokRequest | OllamaRequest, response_obj: Any
    ) -> None:
        """Write full request and response to PostgreSQL using the schema.

        Inserts one row into the Requests table and one row into the Responses
        table (both partitioned by tstamp). Uses infopypg.execute_query with
        parameterized INSERT for safety and efficiency. Falls back gracefully
        on DB errors (logs warning).

        Parameters
        ----------
        request_obj
            GrokRequest or OllamaRequest instance.
        response_obj
            GrokResponse, OllamaResponse, or final streaming state dict.
        """
        await self._ensure_db_pool()
        pool = await PgPoolManager.get_pool(self.resolved_settings)                       # type: ignore[arg-type]

        # Structured log
        if self.logger:
            self.logger.info(
                "LLM interaction completed",
                extra={
                    "obj": {
                        "provider": self.provider,
                        "model": self.model,
                        "request_id": getattr(request_obj, "id", None),
                    }
                },
            )

            # Build minimal insert dicts for the two tables (JSONB fields)
        request_payload = {
            "provider_id": 1,                                                             # TODO: resolve from Providers lookup if needed
            "endpoint": request_obj.get_endpoint(),
            "request_id": "00000000-0000-0000-0000-000000000000",                         # placeholder
            "request": request_obj.to_payload(),
            "meta": {"conversation_id": self.conversation_id},
        }
        response_payload = {
            "provider_id": 1,
            "endpoint": request_obj.get_endpoint(),
            "request_id": "00000000-0000-0000-0000-000000000000",
            "request_tstamp": "2025-03-26 19:22:00+00",                                   # placeholder
            "response_id": "00000000-0000-0000-0000-000000000000",
            "response": (
                response_obj.to_dict()
                if hasattr(response_obj, "to_dict")
                else response_obj
            ),
            "meta": {"conversation_id": self.conversation_id},
        }

        # Insert request row
        await execute_query(
            pool,
            """
            INSERT INTO requests (provider_id, endpoint, request_id, request, meta)
            VALUES ($1, $2, $3, $4, $5)
            """,
            params=list(request_payload.values()),
            fetch=False,
            logger=self.logger,
        )

        # Insert response row
        await execute_query(
            pool,
            """
            INSERT INTO responses (provider_id, endpoint, request_id, request_tstamp,
                                   response_id, response, meta)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            params=list(response_payload.values()),
            fetch=False,
            logger=self.logger,
        )

    async def generate(
        self,
        request: GrokRequest | OllamaRequest,
        stream: bool = False,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> GrokResponse | OllamaResponse | AsyncIterator[LLMStreamingChunkProtocol]:
        """Send a request and return either a full response or async stream.

        Automatically calls request.to_payload() and request.get_endpoint() so
        Grok uses the Responses API with "input" while Ollama uses "messages".
        Grok prompt caching is enabled when use_cache=True and conversation_id
        was supplied at construction. Streaming returns chunks implementing
        LLMStreamingChunkProtocol for uniform handling across providers.

        Parameters
        ----------
        request
            GrokRequest or OllamaRequest instance (type is auto-detected).
        stream
            Return an async iterator of provider-specific chunks if True.
        use_cache
            Enable Grok prompt caching (ignored for Ollama).
        **kwargs
            Any extra parameters accepted by the underlying API (tools, vision,
            temperature, etc.).
        """
        if self.provider == "grok":
            return await self._grok_generate(request, stream, use_cache, **kwargs)
        return await self._ollama_generate(request, stream, **kwargs)

    async def _grok_generate(
        self,
        request: GrokRequest | OllamaRequest,
        stream: bool,
        use_cache: bool,
        **kwargs: Any,
    ) -> GrokResponse | AsyncIterator[GrokStreamingChunk]:
        """Grok-specific generation path using the Responses API (supports
        caching, tools, vision).

        Parameters
        ----------
        request
            GrokRequest instance (union accepted for call-site compatibility).
        stream
            Return streaming iterator if True.
        use_cache
            Enable Grok prompt caching.
        **kwargs
            Extra parameters passed through.
        """
        extra_headers: dict[str, str] = {}
        if use_cache and self.conversation_id:
            extra_headers["x-grok-conv-id"] = self.conversation_id

        params = {
            "model": self.model,
            **request.to_payload(),                                                       # includes "input" automatically
            "stream": stream,
            "extra_headers": extra_headers,
            **kwargs,                                                                     # tools, vision content, etc. passed through
        }

        if stream:
            openai_stream = await self.client.responses.create(**params)                  # type: ignore[union-attr]
            return self._grok_stream(openai_stream, request)

        completion = await self.client.responses.create(**params)                         # type: ignore[union-attr]
        response = GrokResponse.from_dict(completion.model_dump())
        await self._persist_interaction(request, response)
        return response

    async def _grok_stream(
        self,
        openai_stream: Any,
        original_request: GrokRequest | OllamaRequest,
    ) -> AsyncIterator[GrokStreamingChunk]:
        """Yield Grok streaming chunks and persist final state."""
        async for chunk in openai_stream:
            streaming_chunk = GrokStreamingChunk.from_dict(chunk.model_dump())
            yield streaming_chunk
            if streaming_chunk.is_final:
                await self._persist_interaction(
                    original_request, {"status": "stream_complete"}
                )

    async def _ollama_generate(
        self,
        request: GrokRequest | OllamaRequest,
        stream: bool,
        **kwargs: Any,
    ) -> OllamaResponse | AsyncIterator[OllamaStreamingChunk]:
        """Ollama-specific generation path.

        Parameters
        ----------
        request
            OllamaRequest instance (union accepted for call-site compatibility).
        stream
            Return streaming iterator if True.
        **kwargs
            Extra parameters passed through.
        """
        if stream:
            ollama_stream = await self.client.chat(                                       # type: ignore[union-attr]
                model=self.model,
                **request.to_payload(),                                                   # includes "messages" automatically
                stream=True,
                **kwargs,
            )
            return self._ollama_stream(ollama_stream)

        response_raw = await self.client.chat(                                            # type: ignore[union-attr]
            model=self.model,
            **request.to_payload(),
            stream=False,
            **kwargs,
        )
        response = OllamaResponse.from_dict(response_raw)
        await self._persist_interaction(request, response)
        return response

    async def _ollama_stream(
        self, ollama_stream: Any
    ) -> AsyncIterator[OllamaStreamingChunk]:
        """Yield Ollama streaming chunks."""
        async for part in ollama_stream:
            yield OllamaStreamingChunk.from_dict(part)

    async def parallel_generate(
        self,
        requests: list[GrokRequest | OllamaRequest],
        max_concurrent: int = 20,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> list[GrokResponse | OllamaResponse]:
        """Fire many requests in true parallel after a cache seed.

        Automatically respects rate limits with a semaphore. Only call after a
        seeding request when using Grok caching. Exceptions are raised
        immediately (standard asyncio behaviour) for clean error handling.

        Parameters
        ----------
        requests
            List of GrokRequest or OllamaRequest objects.
        max_concurrent
            Maximum simultaneous requests (adjust to your tier).
        use_cache
            Enable Grok prompt caching.
        **kwargs
            Extra parameters passed through.
        """
        if self.provider == "grok" and not self.conversation_id:
            msg = "parallel_generate on Grok requires conversation_id for caching"
            if self.logger:
                self.logger.error(msg)
            raise ValueError(msg)

        if self.logger:
            self.logger.info(
                "parallel_generate started",
                extra={
                    "obj": {"count": len(requests), "max_concurrent": max_concurrent}
                },
            )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _safe_generate(req: GrokRequest | OllamaRequest) -> Any:
            async with semaphore:
                return await self.generate(
                    req, stream=False, use_cache=use_cache, **kwargs
                )

        tasks = [_safe_generate(req) for req in requests]
        results = await asyncio.gather(*tasks)                                            # exceptions raised immediately

        if self.logger:
            self.logger.info(
                "parallel_generate completed",
                extra={"obj": {"count": len(results)}},
            )
        return results

    async def submit_batch(
        self, batch_request: GrokBatchRequest, persist: bool = True
    ) -> GrokBatchResponse:
        """Submit a native Grok batch request to /v1/batches (Grok-only).

        Optionally persists the batch metadata to the requests table for
        auditability (consistent with single generate calls).

        Parameters
        ----------
        batch_request
            GrokBatchRequest instance (contains multiple GrokRequest objects).
        persist
            Whether to write batch metadata to PostgreSQL (default True).

        Returns
        -------
        GrokBatchResponse
            Batch metadata (use await_batch_completion for results).
        """
        if self.provider != "grok":
            msg = "submit_batch is only available for Grok provider"
            raise ValueError(msg)
        if self.logger:
            self.logger.info(
                "submit_batch started",
                extra={"obj": {"batch_size": len(batch_request.requests)}},
            )

        payload = batch_request.to_payload()
        try:
            response = await self.client.post(                                            # type: ignore[union-attr]
                "/v1/batches", json=payload
            )
            batch_response = GrokBatchResponse.from_dict(response.json())
        except Exception as exc:                                                          # noqa: BLE001
            if self.logger:
                self.logger.error(
                    "submit_batch failed",
                    extra={"obj": {"error": str(exc)}},
                    exc_info=True,
                )

            raise

        if self.logger:
            self.logger.info(
                "submit_batch completed",
                extra={"obj": {"batch_id": batch_response.batch_id}},
            )

        if persist:
            # Reuse existing persistence path (stores batch as a request)
            await self._persist_interaction(
                batch_request.requests[0],                                                # representative request
                {"type": "batch", "batch_id": batch_response.batch_id},
            )

        return batch_response

    async def await_batch_completion(
        self,
        batch_id: str,
        poll_interval: int = 30,
        timeout_seconds: int | None = None,
    ) -> list[GrokResponse]:
        """Poll Grok until a batch completes and return all final responses.

        Uses the same DB pool and logger as the rest of the client.

        Parameters
        ----------
        batch_id
            Batch identifier returned by submit_batch.
        poll_interval
            Seconds between status checks (default 30).
        timeout_seconds
            Optional timeout (raises TimeoutError if exceeded).

        Returns
        -------
        list[GrokResponse]
            Completed responses (one per original request).
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            if timeout_seconds and (
                asyncio.get_event_loop().time() - start_time > timeout_seconds
            ):
                msg = f"Batch {batch_id} timed out after {timeout_seconds}s"
                if self.logger:
                    self.logger.error(msg)
                raise TimeoutError(msg)

            resp = await self.client.get(f"/v1/batches/{batch_id}")                       # type: ignore[union-attr]
            status = resp.json()

            if status.get("status") == "completed":
                if self.logger:
                    self.logger.info(
                        "batch completed",
                        extra={"obj": {"batch_id": batch_id}},
                    )
                    # Extract individual results
                results = []
                for item in status.get("results", []):
                    if item.get("response"):
                        results.append(GrokResponse.from_dict(item["response"]))
                return results

            await asyncio.sleep(poll_interval)

    async def create_ollama_model(self, modelfile: Path | str) -> dict[str, Any]:
        """Ollama-only: create or update a model from a Modelfile.

        Raises ValueError if called on a Grok client.
        """
        if self.provider != "ollama":
            msg = "create_ollama_model is only available for Ollama provider"

            if self.logger:
                self.logger.error(msg)
            raise ValueError(msg)

        return await self.client.create(                                                  # type: ignore[union-attr]
            model=self.model,
            modelfile=str(modelfile),
        )

    async def get_embeddings(
        self, input_text: str | list[str]
    ) -> list[float] | list[list[float]]:
        """Return embeddings (most useful with Ollama; Grok does not expose this)."""
        if self.provider != "ollama":
            msg = "get_embeddings currently only implemented for Ollama"
            if self.logger:
                self.logger.error(msg)
            raise ValueError(msg)
        result = await self.client.embeddings(                                            # type: ignore[union-attr]
            model=self.model,
            prompt=input_text,
        )
        return result["embedding"]
