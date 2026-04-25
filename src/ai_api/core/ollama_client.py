"""
High-level asynchronous client for Ollama (local) using native API.

This module provides the public-facing ``OllamaClient`` factory and the four
mode-specific client classes (``TurnOllamaClient``, ``StreamOllamaClient``,
``BatchOllamaClient``, ``EmbedOllamaClient``). It mirrors the exact public
API of ``xai_client.py`` so you can swap providers with minimal code changes.

High-level responsibilities
---------------------------
- Expose a consistent ``create_chat(messages, model, mode=..., response_model=...,
    save_mode=...)`` API.
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
- Imports concrete implementations from ``ollama/chat_*.py``
  and ``ollama/embeddings_ollama.py``.
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
        mode="stream",  # or "turn", "batch"
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

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal, Optional, Type, cast

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
from .client_factory import register_provider
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
    """Shared base class providing HTTP client lifecycle and configuration.

    All concrete Ollama clients (Turn, Stream, Batch, Embed) inherit from this
    class.  It manages a single ``httpx.AsyncClient`` instance (connection
    pooling, timeouts, limits) and holds the optional ``PersistenceManager``.

    Subclasses only need to implement ``create_chat`` (or ``embeddings``);
    the base class takes care of HTTP setup, logging, and graceful shutdown.
    """

    def __init__(
        self,
        logger: logging.Logger,
        host: str = "http://localhost:11434",
        timeout: Optional[int] = 180,
        persistence_manager: "PersistenceManager | None" = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the base Ollama client.

        Parameters
        ----------
        logger : logging.Logger
            Structured logger used by all chat/embedding methods.
        host : str, default "http://localhost:11434"
            Base URL of the Ollama server (trailing slash is stripped).
        timeout : int, optional
            Request timeout in seconds.  Ollama can be slow on large models
            or long generations; the default (180 s) is generous.
        persistence_manager : PersistenceManager, optional
            If supplied, every interaction is persisted via
            ``persist_chat_turn`` (chat turns) or the same method with
            ``kind="embedding"`` (embeddings).
        **kwargs
            Reserved for future extension (e.g. ``verify_ssl``, ``proxy``).
        """
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.logger = logger
        self.persistence_manager = persistence_manager
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Lazily create (and cache) a shared async HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.host,
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._http_client

    async def aclose(self) -> None:
        """Close the underlying HTTP client and release connections."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def get_model_options(self, model: str) -> dict[str, Any]:
        """Fetch the model's default generation parameters from Ollama (/api/show).

        Useful for discovering what defaults a Modelfile ships with before
        overriding them with explicit ``temperature``, ``num_ctx``, etc.

        Returns
        -------
        dict
            The ``parameters`` section of the model manifest (may be a
            string or dict depending on Ollama version).
        """
        http_client = await self._get_http_client()
        resp = await http_client.post("/api/show", json={"model": model})
        resp.raise_for_status()
        data = resp.json()
        return data.get("parameters") or {}

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
                    async with http_client.stream(
                        "POST", "/api/pull", json=payload
                    ) as resp:
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
    """Non-streaming (turn-based) chat client for Ollama.

    This is the workhorse for the majority of chat use cases.  It supports
    structured JSON output, multimodal messages, tools, and full branching
    metadata (passed through to ``persist_chat_turn``).
    """

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
        """Create a single non-streaming chat turn (delegates to create_turn_chat_session).

        All persistence, structured-output handling, and branching metadata
        are managed by the delegated function; this method is intentionally
        thin.
        """
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
    """Real-time streaming chat client for Ollama.

    Yields ``OllamaStreamingChunk`` objects as tokens arrive.  The final
    accumulated response is persisted exactly once via ``persist_chat_turn``
    after the stream completes.
    """

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
        """Start a streaming chat and yield chunks (delegates to generate_stream_and_persist)."""
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
            **kwargs,                                                                     # tree/branch/parent/sequence
        ):
            yield chunk


class BatchOllamaClient(BaseOllamaClient):
    """Batch chat client (simulated – Ollama has no native batch endpoint).

    Runs multiple independent turn-based chats.  Use ``concurrent=True`` only
    when you have sufficient GPU memory or multiple Ollama instances.
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
        """Execute one or more chat turns (delegates to TurnOllamaClient).

        When ``messages`` is a list of lists, each inner list is treated as
        an independent conversation.  Results are returned in the same order.
        """
        turn_client = TurnOllamaClient(
            logger=self.logger,
            host=self.host,
            timeout=self.timeout,
            persistence_manager=self.persistence_manager,
        )

        if isinstance(messages[0], dict):                                                 # single conversation
            return await turn_client.create_chat(
                messages,
                model,
                temperature=temperature,
                max_tokens=max_tokens,
                save_mode=save_mode,
                **kwargs,
            )

        # list of conversations
        if concurrent:
            tasks = [
                turn_client.create_chat(
                    m,
                    model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    save_mode=save_mode,
                    **kwargs,
                )
                for m in messages
            ]
            return await asyncio.gather(*tasks)
        else:
            results: list[OllamaResponse] = []
            for m in messages:
                results.append(
                    await turn_client.create_chat(
                        m,
                        model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        save_mode=save_mode,
                        **kwargs,
                    )
                )
            return results


class EmbedOllamaClient(BaseOllamaClient):
    """Embeddings-only client for Ollama."""

    async def embeddings(
        self,
        input: str | list[str],
        model: str = "nomic-embed-text",
        *,
        save_mode: SaveMode = "none",
        **kwargs: Any,
    ) -> OllamaEmbedResponse:
        """Generate embeddings

        (delegates to create_embeddings with unified persistence)."""
        return await create_embeddings(
            self, input, model, save_mode=save_mode, **kwargs
        )


# Convenience factory (used by client_factory.py)
def OllamaClient(
    logger: logging.Logger,
    mode: ChatMode = "turn",
    host: str = "http://localhost:11434",
    timeout: Optional[int] = 180,
    persistence_manager: "PersistenceManager | None" = None,
    **kwargs: Any,
) -> TurnOllamaClient | StreamOllamaClient | BatchOllamaClient | EmbedOllamaClient:
    """Factory returning the appropriate Ollama client for the requested mode.

    This is the function registered with ``client_factory.register_provider``.
    End users normally obtain instances via ``get_llm_client("ollama", ...)``.
    """
    if mode == "turn":
        return TurnOllamaClient(
            logger,
            host=host,
            timeout=timeout,
            persistence_manager=persistence_manager,
            **kwargs,
        )
    elif mode == "stream":
        return StreamOllamaClient(
            logger,
            host=host,
            timeout=timeout,
            persistence_manager=persistence_manager,
            **kwargs,
        )
    elif mode == "batch":
        return BatchOllamaClient(
            logger,
            host=host,
            timeout=timeout,
            persistence_manager=persistence_manager,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported Ollama mode: {mode}")


# Auto-register with the central factory
register_provider("ollama", TurnOllamaClient, StreamOllamaClient, BatchOllamaClient)
