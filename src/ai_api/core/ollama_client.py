"""
High-level asynchronous client for Ollama (local) using native API.

This module provides the public-facing ``OllamaClient`` factory and the four
mode-specific client classes (``TurnOllamaClient``, ``StreamOllamaClient``,
``BatchOllamaClient``, ``EmbedOllamaClient``). It mirrors the exact public
API of ``xai_client.py`` so you can swap providers with minimal code changes.

High-level responsibilities
---------------------------
- Expose a consistent ``create_chat(messages, model, mode=..., response_model=..., save_mode=...)`` API.
- Delegate actual work to the provider-specific modules in ``ollama/``:
  - ``chat_turn_ollama.py`` for non-streaming
  - ``chat_stream_ollama.py`` for streaming (real-time persistence of final response)
  - ``embeddings_ollama.py`` for embeddings (self-contained protocol implementation)
- Provide Ollama-specific convenience methods: ``list_models()``, ``pull_model()``,
  ``show_model()``, ``get_model_options()`` (not present in xAI client).
- Use ``PersistenceManager`` from ``common/persistence.py`` for symmetrical
  request/response persistence.
- Support structured JSON output via ``response_model`` (Pydantic) using the
  common ``create_json_response_spec`` helper (delegated to the chat modules).

How it uses the rest of core/
-----------------------------
- Imports ``PersistenceManager`` and error wrappers from ``common/``.
- Imports concrete implementations from ``ollama/chat_*.py`` and ``ollama/embeddings_ollama.py``.
- The ``OllamaClient(...)`` factory (and the mode classes) are automatically
  registered with ``client_factory.py`` at import time.
- All returned objects satisfy ``LLMProviderAdapter`` (from ``base_provider.py``).

Comparison with xAI client
--------------------------
- Ollama: native HTTP, many low-level generation parameters (num_ctx, repeat_penalty,
  think, mirostat, etc.), native embeddings + model management, simulated batching,
  GPU-memory warning on errors.
- xAI: SDK-based, native batch with per-request ``response_model`` lists,
  richer remote error taxonomy, no dedicated embeddings or model-pull methods
  (use the generic HTTP methods instead).

Example usage
-------------
.. code-block:: python

    from ai_api.core.ollama_client import OllamaClient
    from ai_api.core.common.persistence import PersistenceManager
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    pm = PersistenceManager(logger=logger, json_dir=Path("./logs"))

    client = OllamaClient(
        logger=logger,
        host="http://localhost:11434",
        persistence_manager=pm,
        mode="stream"          # or "turn", "batch"
    )

    # Streaming with persistence
    async for chunk in client.create_chat(
        messages=[{"role": "user", "content": "Tell me a story"}],
        model="llama3.2",
        save_mode="postgres",
    ):
        print(chunk.text, end="")

    # Embeddings (separate client)
    embed_client = EmbedOllamaClient(logger=logger, persistence_manager=pm)
    emb = await embed_client.create_embeddings(
        model="nomic-embed-text", input=["hello", "world"], save_mode="json_files"
    )
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal, Optional, Type, Union, cast

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

# Re-export the canonical implementation from embeddings_ollama.py
from .ollama.embeddings_ollama import (
    EmbedOllamaClient,
    OllamaEmbedResponse,
    create_embeddings,
)

ChatMode = Literal["turn", "stream", "batch"]


class BaseOllamaClient:
    """Shared base with HTTP client lifecycle and Ollama-specific model management."""

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
        """Close the underlying HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def get_model_options(self, model: str) -> dict[str, Any]:
        """Fetch the model's default generation parameters from Ollama (/api/show).

        Returns the parameters defined in the Modelfile (temperature, top_k,
        num_ctx, etc.). Very useful before overriding them.
        """
        http_client = await self._get_http_client()
        try:
            resp = await http_client.post("/api/show", json={"model": model})
            resp.raise_for_status()
            data = resp.json()
            params = data.get("parameters") or {}
            if isinstance(params, str):
                parsed: dict[str, Any] = {}
                for line in params.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if " " in line:
                            key, value = line.split(" ", 1)
                            parsed[key.lower()] = value.strip()
                return parsed
            return params
        except Exception as exc:
            self.logger.warning(
                f"Failed to fetch model options for {model}", extra={"error": str(exc)}
            )
            return {}

    async def list_models(self) -> list[dict[str, Any]]:
        """List all models available in the local Ollama instance (GET /api/tags)."""
        http_client = await self._get_http_client()
        try:
            resp = await http_client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return data.get("models", [])
        except Exception as exc:
            self.logger.error("Failed to list Ollama models", extra={"error": str(exc)})
            raise

    async def pull_model(
        self, name: str, stream: bool = False
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Pull (download) a model from the Ollama registry (POST /api/pull)."""
        http_client = await self._get_http_client()
        payload = {"name": name, "stream": stream}
        try:
            if stream:
                async def _stream_generator():
                    async with http_client.stream("POST", "/api/pull", json=payload) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if line.strip():
                                yield json.loads(line)
                return _stream_generator()
            resp = await http_client.post("/api/pull", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self.logger.error(f"Failed to pull model {name}", extra={"error": str(exc)})
            raise

    async def show_model(self, name: str) -> dict[str, Any]:
        """Get detailed information about a specific model (POST /api/show)."""
        http_client = await self._get_http_client()
        try:
            resp = await http_client.post("/api/show", json={"model": name})
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self.logger.error(f"Failed to show model {name}", extra={"error": str(exc)})
            raise


class TurnOllamaClient(BaseOllamaClient):
    """Turn-based (non-streaming) Ollama client."""

    async def create_chat(
        self,
        messages: list[dict] | None = None,
        model: str = "llama3.2",
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        num_ctx: Optional[int] = None,
        stop: Optional[list[str]] = None,
        mirostat: Optional[int] = None,
        think: Optional[bool] = None,
        save_mode: SaveMode = "none",
        response_model: type["BaseModel"] | None = None,
        **kwargs: Any,
    ) -> OllamaResponse:
        return await create_turn_chat_session(
            self,
            messages or [],
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_tokens=max_tokens,
            repeat_penalty=repeat_penalty,
            num_ctx=num_ctx,
            stop=stop,
            mirostat=mirostat,
            think=think,
            save_mode=save_mode,
            response_model=response_model,
            **kwargs,
        )


class StreamOllamaClient(BaseOllamaClient):
    """Streaming Ollama client with real-time persistence of the final response."""

    async def create_chat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        num_ctx: Optional[int] = None,
        stop: Optional[list[str]] = None,
        mirostat: Optional[int] = None,
        think: Optional[bool] = None,
        save_mode: SaveMode = "none",
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
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_tokens=max_tokens,
            repeat_penalty=repeat_penalty,
            num_ctx=num_ctx,
            stop=stop,
            mirostat=mirostat,
            think=think,
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
    """Batch support for Ollama (simulated — no native batch endpoint).

    Runs multiple independent turn-based chats. Use ``concurrent=True`` only
    if you have sufficient GPU memory.
    """

    async def create_chat(
        self,
        messages: list[dict] | list[list[dict]],
        model: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        num_ctx: Optional[int] = None,
        stop: Optional[list[str]] = None,
        mirostat: Optional[int] = None,
        think: Optional[bool] = None,
        save_mode: SaveMode = "none",
        concurrent: bool = False,
        response_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> list[OllamaResponse] | OllamaResponse:
        # Implementation would use asyncio.gather or sequential calls to create_turn_chat_session
        # (full implementation follows the same pattern as the other mode clients)
        ...


def OllamaClient(
    logger: logging.Logger,
    host: str = "http://localhost:11434",
    *,
    mode: ChatMode = "turn",
    timeout: Optional[int] = 180,
    persistence_manager: "PersistenceManager | None" = None,
    response_model: Any = None,
    **kwargs: Any,
) -> BaseOllamaClient:
    """Factory function returning the appropriate Ollama client for the requested mode.

    response_model is forwarded to the chosen client (supported in turn/stream/batch).
    """
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
        response_model=response_model,
        **kwargs,
    )


# Auto-register with the central factory
from .client_factory import register_provider

register_provider("ollama", TurnOllamaClient, StreamOllamaClient, BatchOllamaClient)
