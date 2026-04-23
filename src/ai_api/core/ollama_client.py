"""
ollama_client.py — fully updated with unified `response_model` support.

Now supports structured output (Pydantic models) consistently across:
- Turn mode
- Stream mode
- Batch mode (single model or per-request list of models)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Literal, Optional, Type

import httpx
from pydantic import BaseModel

from ..data_structures.ollama_objects import (
    LLMStreamingChunkProtocol,
    OllamaInput,
    OllamaJSONResponseSpec,
    OllamaMessage,
    OllamaRequest,
    OllamaResponse,
    OllamaRole,
    OllamaStreamingChunk,
    SaveMode,
)
from .common.persistence import PersistenceManager
from .ollama.chat_stream_ollama import generate_stream_and_persist
from .ollama.chat_turn_ollama import create_turn_chat_session
from .ollama.embeddings_ollama import EmbedOllamaClient, OllamaEmbedResponse

ChatMode = Literal["turn", "stream", "batch"]


class BaseOllamaClient:
    """Shared base with HTTP client lifecycle."""

    def __init__(
        self,
        logger: logging.Logger,
        host: str = "http://localhost:11434",
        timeout: Optional[int] = 180,
        persistence_manager: "PersistenceManager | None" = None,
        **kwargs: Any,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.logger = logger
        self.persistence_manager = persistence_manager
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.host,
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._http_client

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


class TurnOllamaClient(BaseOllamaClient):
    async def create_chat(
        self,
        messages: list[dict] | None = None,
        model: str = "llama3.2",
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_mode: SaveMode = "none",
        response_model: Type[BaseModel] | None = None,                                    # ← NEW
        **kwargs: Any,
    ) -> OllamaResponse:
        return await create_turn_chat_session(
            self,
            messages or [],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            response_model=response_model,                                                # ← PASS THROUGH
            **kwargs,
        )


class StreamOllamaClient(BaseOllamaClient):
    async def create_chat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_mode: SaveMode = "none",
        response_model: Type[BaseModel] | None = None,                                    # ← NEW
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamingChunkProtocol]:
        self.logger.info(
            "Starting Ollama streaming chat",
            extra={"model": model, "save_mode": save_mode},
        )

        ollama_input = OllamaInput.from_list(messages)
        request = OllamaRequest(
            model=model,
            input=ollama_input,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            **kwargs,
        )

        http_client = await self._get_http_client()

        async for chunk in generate_stream_and_persist(
            self.logger,
            self.persistence_manager,
            http_client,
            request,
            save_mode=save_mode,
            response_model=response_model,                                                # ← PASS THROUGH
        ):
            yield chunk


class BatchOllamaClient(BaseOllamaClient):
    """Batch support for Ollama (simulated – now supports flexible response_model)."""

    async def create_chat(
        self,
        messages: list[list[dict]],
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_mode: SaveMode = "none",
        response_model: list[Type[BaseModel]]
        | Type[BaseModel]
        | None = None,                                                                    # ← FLEXIBLE
        **kwargs: Any,
    ) -> list[OllamaResponse]:
        """Simulated batch with per-request or shared structured output."""
        self.logger.info("Ollama batch chat (simulated)", extra={"model": model})

        results = []
        for idx, msgs in enumerate(messages):
            # Determine response_model for this specific request
            if isinstance(response_model, list):
                current_rm = response_model[idx]
            else:
                current_rm = response_model

            turn_client = TurnOllamaClient(
                logger=self.logger,
                host=self.host,
                timeout=self.timeout,
                persistence_manager=self.persistence_manager,
            )
            result = await turn_client.create_chat(
                messages=msgs,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                save_mode=save_mode,
                response_model=current_rm,
                **kwargs,
            )
            results.append(result)

        return results


def OllamaClient(
    logger: logging.Logger,
    host: str = "http://localhost:11434",
    *,
    mode: ChatMode = "turn",
    timeout: Optional[int] = 180,
    persistence_manager: "PersistenceManager | None" = None,
    **kwargs: Any,
) -> BaseOllamaClient:
    """Factory – now fully supports response_model across all modes."""
    client_map: dict[ChatMode, Type[BaseOllamaClient]] = {
        "turn": TurnOllamaClient,
        "stream": StreamOllamaClient,
        "batch": BatchOllamaClient,
    }

    ClientClass = client_map.get(mode)
    if ClientClass is None:
        raise ValueError(
            f"Unsupported mode '{mode}'. Must be one of: {list(client_map.keys())}"
        )

    return ClientClass(
        logger=logger,
        host=host,
        timeout=timeout,
        persistence_manager=persistence_manager,
        **kwargs,
    )


class EmbedOllamaClient(BaseOllamaClient):
    async def create_embeddings(self, *args, **kwargs) -> "OllamaEmbedResponse":
        from .ollama.embeddings_ollama import EmbedOllamaClient as EmbedImpl
        from .ollama.embeddings_ollama import create_embeddings

        impl = EmbedImpl(
            logger=self.logger,
            host=self.host,
            timeout=self.timeout,
            persistence_manager=self.persistence_manager,
        )
        return await create_embeddings(impl, *args, **kwargs)
