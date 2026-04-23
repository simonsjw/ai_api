"""
High-level asynchronous client for xAI (remote) using the official SDK.

This module provides the public-facing ``XAIClient`` factory and the three
mode-specific client classes (``TurnXAIClient``, ``StreamXAIClient``,
``BatchXAIClient``). It mirrors the public API of ``ollama_client.py`` so
providers are interchangeable.

High-level responsibilities
---------------------------
- Expose a consistent ``create_chat(messages, model, mode=..., response_model=..., save_mode=...)`` API.
- Delegate to provider-specific modules in ``xai/``:
  - ``chat_turn_xai.py`` for non-streaming
  - ``chat_stream_xai.py`` for streaming
  - ``chat_batch_xai.py`` for native batch (unique to xAI — supports per-request ``response_model`` lists)
- Use ``PersistenceManager`` from ``common/persistence.py`` for symmetrical persistence.
- Support structured JSON output via ``response_model`` (Pydantic) using the common helper.
- Provide xAI-specific methods: ``list_models()``, ``get_model_info()``.

How it uses the rest of core/
-----------------------------
- Imports ``PersistenceManager`` and error wrappers from ``common/``.
- Imports concrete implementations from ``xai/chat_*.py``.
- The ``XAIClient(...)`` factory is automatically registered with
  ``client_factory.py`` at import time.
- All returned objects satisfy ``LLMProviderAdapter`` (from ``base_provider.py``).

Comparison with Ollama client
-----------------------------
- xAI: SDK-based, native batch with flexible per-request structured output,
  richer remote error taxonomy (rate limits, thinking mode, multimodal, cache).
- Ollama: native HTTP, many low-level generation parameters, native embeddings
  + model management, simulated batching, GPU-memory warning.

Example usage
-------------
.. code-block:: python

    from ai_api.core.xai_client import XAIClient
    from ai_api.core.common.persistence import PersistenceManager
    import logging

    logger = logging.getLogger(__name__)
    pm = PersistenceManager(logger=logger, db_url=...)

    client = XAIClient(
        logger=logger,
        api_key="xai-...",
        mode="batch",
        persistence_manager=pm
    )

    results = await client.create_chat(
        messages_list=[conv1, conv2, conv3],
        model="grok-4",
        response_model=[Person, Summary, None],   # different model per request
        save_mode="postgres"
    )
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Literal, Optional, Type

import httpx
from pydantic import BaseModel
from xai_sdk import AsyncClient as XAIAsyncClient

from ..data_structures.xai_objects import (
    LLMStreamingChunkProtocol,
    SaveMode,
    xAIInput,
    xAIRequest,
)
from .common.persistence import PersistenceManager
from .xai.chat_batch_xai import create_batch_chat
from .xai.chat_stream_xai import generate_stream_and_persist
from .xai.chat_turn_xai import create_turn_chat_session

ChatMode = Literal["turn", "stream", "batch"]


class BaseXAIClient:
    """Shared base class containing SDK + HTTP client lifecycle logic."""

    def __init__(
        self,
        logger: logging.Logger,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        timeout: Optional[int] = 120,
        persistence_manager: "PersistenceManager" | None = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.logger = logger
        self.persistence_manager = persistence_manager
        self._http_client: httpx.AsyncClient | None = None
        self._sdk_client: XAIAsyncClient | None = None

    async def _get_sdk_client(self) -> XAIAsyncClient:
        if self._sdk_client is None:
            self._sdk_client = XAIAsyncClient(api_key=self.api_key)
        return self._sdk_client

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        return self._http_client

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def list_models(self) -> list[dict[str, Any]]:
        """List all available Grok models via the xAI API (/v1/models)."""
        http_client = await self._get_http_client()
        try:
            resp = await http_client.get("/v1/models")
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception as exc:
            self.logger.warning(
                "Failed to list xAI models via HTTP", extra={"error": str(exc)}
            )
            return [
                {"id": "grok-4", "created": 1730000000, "owned_by": "xai"},
                {"id": "grok-3", "created": 1725000000, "owned_by": "xai"},
                {"id": "grok-2", "created": 1720000000, "owned_by": "xai"},
            ]

    async def get_model_info(self, model: str) -> dict[str, Any]:
        """Get detailed information about a specific Grok model."""
        http_client = await self._get_http_client()
        try:
            resp = await http_client.get(f"/v1/models/{model}")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            all_models = await self.list_models()
            for m in all_models:
                if m.get("id") == model:
                    return m
            self.logger.warning(f"Model '{model}' not found in xAI catalog")
            return {"id": model, "error": "Model not found or details unavailable"}


class TurnXAIClient(BaseXAIClient):
    async def create_chat(
        self,
        messages: list[dict] | None = None,
        model: str = "grok-4",
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_mode: SaveMode = "none",
        response_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._sdk_client is None:
            self._sdk_client = XAIAsyncClient(api_key=self.api_key)
        return await create_turn_chat_session(
            self,
            self._sdk_client,
            messages or [],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            response_model=response_model,
            **kwargs,
        )


class StreamXAIClient(BaseXAIClient):
    async def create_chat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_mode: SaveMode = "none",
        response_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamingChunkProtocol]:
        sdk_client = await self._get_sdk_client()

        xai_input = xAIInput.from_list(messages)
        request = xAIRequest(
            input=xai_input,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            **kwargs,
        )

        chat = sdk_client.chat.create(
            model=request.model,
            **request.to_sdk_chat_kwargs(),
        )

        async for chunk in generate_stream_and_persist(
            self.logger,
            self.persistence_manager,
            chat,
            request,
            save_mode=save_mode,
            response_model=response_model,
        ):
            yield chunk


class BatchXAIClient(BaseXAIClient):
    async def create_chat(
        self,
        messages: list[list[dict]],
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_mode: SaveMode = "none",
        response_model: list[Type[BaseModel]] | Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> Any:
        return await create_batch_chat(
            self,
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            response_model=response_model,
            **kwargs,
        )


def XAIClient(
    logger: logging.Logger,
    api_key: str,
    *,
    mode: ChatMode = "turn",
    base_url: str = "https://api.x.ai/v1",
    timeout: Optional[int] = 120,
    persistence_manager: "PersistenceManager" | None = None,
    response_model: Any = None,
    **kwargs: Any,
) -> BaseXAIClient:
    """Factory function returning a specialised xAI client.

    response_model is forwarded (single model or list for batch mode).
    """
    client_map: dict[ChatMode, Type[BaseXAIClient]] = {
        "turn": TurnXAIClient,
        "stream": StreamXAIClient,
        "batch": BatchXAIClient,
    }

    ClientClass = client_map.get(mode)
    if ClientClass is None:
        raise ValueError(
            f"Unsupported mode '{mode}'. Must be one of: {list(client_map.keys())}"
        )

    return ClientClass(
        logger=logger,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        persistence_manager=persistence_manager,
        response_model=response_model,
        **kwargs,
    )


# Auto-register with the central factory
from .client_factory import register_provider

register_provider("xai", TurnXAIClient, StreamXAIClient, BatchXAIClient)
