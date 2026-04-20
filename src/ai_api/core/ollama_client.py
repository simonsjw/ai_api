"""High-level asynchronous client for Ollama (local) using native API.

Mirrors the exact public API and behaviour of xai_client.py so you can
swap providers with minimal code changes:

    client = OllamaClient(logger=logger, host="http://localhost:11434", ...)
    response = await client.create_chat(messages=..., model="llama3.2", ...)

Fully supports:
- Turn-based (non-streaming)
- Streaming (with real-time persistence)
- Structured JSON output via OllamaJSONResponseSpec
- Multimodal (base64 images in messages)
- Since Batching is not a functionality provided by Ollama, it is not implemented here.
- Your existing xAIPersistenceManager (reused unchanged)
- Same SaveMode, logging pattern, and error wrapping style
"""

import json
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

ChatMode = Literal["turn", "stream", "batch"]


class BaseOllamaClient:
    """Shared base with HTTP client lifecycle."""

    def __init__(
        self,
        logger: logging.Logger,
        host: str = "http://localhost:11434",
        timeout: Optional[int] = 180,                                                     # Ollama can be slower on large models
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
        """Close the underlying HTTP client."""
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
        response_model: type["BaseModel"] | None = None,
        **kwargs: Any,
    ) -> OllamaResponse:
        return await create_turn_chat_session(
            self,
            messages or [],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            response_model=response_model,
            **kwargs,
        )


class StreamOllamaClient(BaseOllamaClient):
    """Streaming client for Ollama (now thin and fully delegated)."""

    async def create_chat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_mode: SaveMode = "none",
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamingChunkProtocol]:
        """Streaming Ollama chat – delegates persistence + streaming to generate_stream_and_persist."""
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
        ):
            yield chunk


class BatchOllamaClient(BaseOllamaClient):
    """Batch support for Ollama (simulated – runs requests sequentially, as Ollama has no native batch API)."""

    async def create_chat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> OllamaResponse:
        """Simulated batch (single request for now – extendable to true parallel later)."""
        self.logger.info("Ollama batch chat (simulated)", extra={"model": model})
        # For local Ollama we just delegate to turn mode
        turn_client = TurnOllamaClient(
            logger=self.logger,
            host=self.host,
            timeout=self.timeout,
            persistence_manager=self.persistence_manager,
        )
        return await turn_client.create_chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


def OllamaClient(
    logger: logging.Logger,
    host: str = "http://localhost:11434",
    *,
    mode: ChatMode = "turn",
    timeout: Optional[int] = 180,
    persistence_manager: "PersistenceManager | None" = None,
    **kwargs: Any,
) -> BaseOllamaClient:
    """Factory – exactly mirrors XAIClient factory."""
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
