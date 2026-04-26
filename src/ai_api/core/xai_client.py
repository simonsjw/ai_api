"""
High-level asynchronous client for xAI (remote) using the official xAI SDK.

This module provides the public-facing ``XAIClient`` factory and the mode-specific
client classes (``TurnXAIClient``, ``StreamXAIClient``, ``BatchXAIClient``,
``EmbedXAIClient``). It mirrors the public API surface of ``ollama_client.py``
so that providers are interchangeable with minimal (or zero) code changes in
user applications.

High-level responsibilities
---------------------------
- Expose a consistent ``create_chat(messages, model, mode=..., response_model=...,
    save_mode=...)`` API for turn, stream, and native batch modes.
- Delegate actual LLM calls and persistence orchestration to the provider-specific
  modules in ``xai/``:
    - ``chat_turn_xai.py`` for non-streaming single turns (with structured output)
    - ``chat_stream_xai.py`` for real-time token streaming (final response persisted)
    - ``chat_batch_xai.py`` for efficient per-request structured output in batch
- Provide xAI-specific convenience methods: ``list_models()``, ``get_model_info()``.
- Support symmetrical persistence (request + response) via ``PersistenceManager``
  for all modes when ``save_mode != "none"``.
- Full support for Pydantic ``response_model`` (single or per-item list in batch).

How it uses the rest of core/
-----------------------------
- Inherits shared HTTP/SDK client lifecycle from ``BaseXAIClient``.
- Imports concrete implementations from ``xai/chat_*.py``.
- The ``XAIClient(...)`` factory (and mode classes) are automatically registered
  with ``client_factory.py`` at import time via ``register_provider``.
- All returned objects satisfy ``LLMProviderAdapter`` (structural Protocol from
  ``base_provider.py``).
- Re-uses ``create_json_response_spec`` from ``common/response_struct.py`` for
  structured output (already wired inside the delegated functions).

Comparison with Ollama client
-----------------------------
- **xAI**: SDK-based, native batch endpoint with flexible per-request
  ``response_model`` lists, richer remote error taxonomy, thinking mode,
  multimodal, prompt caching. No native embeddings or model management
  (use generic HTTP methods or the ``EmbedXAIClient`` stub for future).
- **Ollama**: native HTTP, dozens of low-level generation parameters
  (num_ctx, repeat_penalty, think, mirostat, ...), native embeddings +
  model pull/show/list, simulated batch via asyncio, GPU-memory warnings on OOM.

Example usage (recommended via factory)
---------------------------------------
.. code-block:: python

    from ai_api.core.client_factory import get_llm_client
    from ai_api.core.common.persistence import PersistenceManager
    from pydantic import BaseModel
    import logging

    logger = logging.getLogger(__name__)
    pm = PersistenceManager(logger=logger, db_url="postgresql://...")

    # Streaming (real-time tokens + final persistence)
    xai_stream = get_llm_client(
        "xai", logger=logger, mode="stream", api_key="xai-...", persistence_manager=pm
    )
    async for chunk in xai_stream.create_chat(
        messages=[{"role": "user", "content": "Explain Grok-4"}],
        model="grok-4",
        save_mode="postgres",
    ):
        print(chunk.text, end="")


    # Native batch with per-item structured output
    class Person(BaseModel):
        name: str
        age: int


    xai_batch = get_llm_client("xai", logger=logger, mode="batch", api_key=...)
    results = await xai_batch.create_chat(
        messages_list=[conv1, conv2, conv3],
        model="grok-4",
        response_model=[Person, None, Person],  # different per request
        save_mode="json_files",
    )

See Also
--------
ai_api.core.ollama_client : Local provider implementation (identical public surface).
ai_api.core.client_factory : Unified entry point (get_llm_client).
ai_api.core.xai.chat_turn_xai, chat_stream_xai, chat_batch_xai : Delegated logic.
ai_api.core.common.persistence.PersistenceManager : Symmetrical request/response storage.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal, Optional, Type, cast

import httpx
from pydantic import BaseModel
from xai_sdk import AsyncClient as XAIAsyncClient

from ..data_structures.xai_objects import (
    LLMStreamingChunkProtocol,
    SaveMode,
    xAIBatchResponse,
    xAIInput,
    xAIRequest,
    xAIResponse,
)
from .common.persistence import PersistenceManager
from .xai.chat_batch_xai import create_batch_chat
from .xai.chat_stream_xai import generate_stream_and_persist
from .xai.chat_turn_xai import create_turn_chat_session

ChatMode = Literal["turn", "stream", "batch"]


class BaseXAIClient:
    """Shared base class for all xAI clients (Turn, Stream, Batch, Embed).

    Manages API key, base URL, timeout, logger, optional PersistenceManager,
    and lazy-initialised SDK + HTTP clients (connection pooling, auth headers).

    Subclasses only implement the mode-specific ``create_chat`` (or ``embeddings``);
    the base class handles client lifecycle, graceful shutdown via ``aclose()``,
    and common xAI methods (``list_models``, ``get_model_info``).

    All subclasses satisfy the ``LLMProviderAdapter`` Protocol automatically
    because they define a compatible ``create_chat`` method.
    """

    def __init__(
        self,
        logger: logging.Logger,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        timeout: Optional[int] = 120,
        persistence_manager: "PersistenceManager | None" = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the base xAI client.

        Parameters
        ----------
        logger : logging.Logger
            Structured logger used by all methods for request/response tracing.
        api_key : str
            xAI API key (required). Must start with ``xai-``.
        base_url : str, default "https://api.x.ai/v1"
            xAI API base URL (trailing slash stripped automatically).
        timeout : int, optional
            Request timeout in seconds. Default 120 s is generous for long
            generations or thinking mode.
        persistence_manager : PersistenceManager, optional
            If supplied, every interaction (turn, stream final response, batch
            items, embeddings) is persisted symmetrically via
            ``persist_request`` / ``persist_response`` (or ``persist_chat_turn``
            for legacy paths).
        **kwargs
            Reserved for future extension (e.g. ``verify_ssl``, ``proxy``,
            ``extra_headers``).
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.logger = logger
        self.persistence_manager = persistence_manager
        self._http_client: httpx.AsyncClient | None = None
        self._sdk_client: XAIAsyncClient | None = None

    async def _get_sdk_client(self) -> XAIAsyncClient:
        """Lazily create (and cache) the official xAI async SDK client."""
        if self._sdk_client is None:
            self._sdk_client = XAIAsyncClient(api_key=self.api_key)
        return self._sdk_client

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Lazily create (and cache) a shared async HTTP client with auth."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"},
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
            )
        return self._http_client

    async def aclose(self) -> None:
        """Close underlying HTTP and SDK clients and release connections."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        # SDK client has no explicit close in current xai_sdk; future-proof
        self._sdk_client = None

    async def list_models(self) -> list[dict[str, Any]]:
        """List all available Grok models via the xAI catalog (GET /v1/models).

        Falls back to a static list on network error for robustness.

        Returns
        -------
        list[dict]
            Model metadata (id, created, owned_by, ...).
        """
        http_client = await self._get_http_client()
        try:
            resp = await http_client.get("/v1/models")
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception as exc:
            self.logger.warning(
                "Failed to list xAI models via HTTP – using fallback catalog",
                extra={"error": str(exc)},
            )
            return [
                {"id": "grok-4", "created": 1730000000, "owned_by": "xai"},
                {"id": "grok-3", "created": 1725000000, "owned_by": "xai"},
                {"id": "grok-2", "created": 1720000000, "owned_by": "xai"},
            ]

    async def get_model_info(self, model: str) -> dict[str, Any]:
        """Get detailed information about a specific Grok model.

        Tries the dedicated endpoint first, then falls back to scanning the
        full catalog.

        Parameters
        ----------
        model : str
            Model identifier (e.g. "grok-4").

        Returns
        -------
        dict
            Model details or {"id": model, "error": "..."} on failure.
        """
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
    """Non-streaming (turn-based) chat client for xAI.

    Supports structured JSON output via Pydantic ``response_model``, full
    multimodal messages, tools, and symmetrical persistence. This is the
    workhorse for most production chat use cases.
    """

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
    ) -> xAIResponse:
        """Create a single non-streaming chat turn (delegates to create_turn_chat_session).

        All request construction, structured-output spec attachment, request
        persistence, SDK call, response parsing, and response persistence are
        handled inside the delegated function. This method is intentionally thin.

        Parameters
        ----------
        messages : list[dict] or None
            OpenAI-compatible chat history. If None, treated as empty list.
        model : str, default "grok-4"
            xAI model name.
        temperature, max_tokens : float/int, optional
            Generation parameters (passed through to xAIRequest).
        save_mode : {"none", "json_files", "postgres"}, default "none"
            Persistence backend.
        response_model : type[BaseModel] or None, optional
            Pydantic model for structured output. Converted internally to
            ``xAIJSONResponseSpec``.
        **kwargs
            Additional parameters forwarded (e.g. top_p, presence_penalty,
            thinking_mode, etc.).

        Returns
        -------
        xAIResponse
            Completed response object (with optional ``.parsed`` attribute
            when ``response_model`` was supplied).
        """
        sdk_client = await self._get_sdk_client()
        return await create_turn_chat_session(
            self,
            sdk_client,
            messages or [],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            response_model=response_model,
            **kwargs,
        )


class StreamXAIClient(BaseXAIClient):
    """Real-time streaming chat client for xAI.

    Yields ``xAIStreamingChunk`` (or raw SDK chunks) as tokens arrive.
    The final assembled response is constructed, optionally validated against
    a ``response_model``, and persisted exactly once via ``persist_response``
    after the stream completes. Request persistence (if any) occurs before
    streaming begins.
    """

    async def create_chat(
        self,
        messages: list[dict[str, Any]],
        model: str = "grok-4",
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_mode: SaveMode = "none",
        response_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamingChunkProtocol]:
        """Start a streaming chat and yield real-time chunks (delegates to generate_stream_and_persist).

        Parameters
        ----------
        messages : list[dict[str, Any]]
            OpenAI-compatible chat history (required for streaming).
        model : str, default "grok-4"
            xAI model name.
        temperature, max_tokens : float/int, optional
            Generation parameters.
        save_mode : {"none", "json_files", "postgres"}, default "none"
            Persistence backend for the *final* response only.
        response_model : type[BaseModel] or None, optional
            Pydantic model for structured output on the final accumulated text.
        **kwargs
            Forwarded to xAIRequest (e.g. top_p, thinking_mode).

        Yields
        ------
        LLMStreamingChunkProtocol
            Real-time token chunks (text, finish_reason, is_final, raw, ...).
        """
        self.logger.info(
            "Starting xAI streaming chat",
            extra={"model": model, "save_mode": save_mode},
        )

        # 1. Build request object (used for context, persistence linking, and SDK kwargs)
        request = xAIRequest(
            model=model,
            input=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            **kwargs,
        )

        # 2. (Request persistence is optional for streaming; final response is persisted
        #    inside generate_stream_and_persist. If a persist_request helper is added
        #    to PersistenceManager later, call it here.)

        # 3. Obtain SDK client and create the streaming chat iterator
        sdk_client = await self._get_sdk_client()
        # Use the same pattern as chat_turn_xai.py for compatibility with current SDK usage
        chat = sdk_client.chat.create(
            model=request.model,
            **request.to_sdk_chat_kwargs(),
        )
        chat_iterator: Any = (
            chat                                                                          # the create result is the async iterator for streaming
        )

        # 4. Delegate streaming + final persistence + optional structured parsing
        async for chunk in generate_stream_and_persist(
            self.logger,
            self.persistence_manager,
            chat_iterator,
            request,
            save_mode=save_mode,
            response_model=response_model,
        ):
            yield cast(LLMStreamingChunkProtocol, chunk)


class BatchXAIClient(BaseXAIClient):
    """Native batch chat client for xAI (supports per-request structured output).

    When ``messages`` is a list of conversations, each item is processed
    (re-using turn logic for consistency). When a single conversation is
    passed, it is wrapped and a single response is returned (API symmetry with
    Ollama batch client).

    Supports three ``response_model`` variants:
    - None → unstructured
    - Single BaseModel → same model for every item
    - list[BaseModel] → different model per item (length must match batch size)
    """

    async def create_chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        model: str = "grok-4",
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_mode: SaveMode = "none",
        response_model: Type[BaseModel] | list[Type[BaseModel]] | None = None,
        concurrent: bool = False,                                                         # accepted for API symmetry, ignored (native batch is remote)
        **kwargs: Any,
    ) -> list[xAIResponse] | xAIResponse:
        """Execute one or more chat turns via xAI batch path (delegates to create_batch_chat).

        Parameters
        ----------
        messages : list[dict] or list[list[dict]]
            Either a single conversation or a list of conversations.
        model : str, default "grok-4"
            xAI model applied to all items.
        temperature, max_tokens : float/int, optional
            Common generation parameters.
        save_mode : {"none", "json_files", "postgres"}, default "none"
            Persistence backend (applied per item).
        response_model : Type[BaseModel], list[Type[BaseModel]], or None, optional
            Structured output specification (see class docstring).
        concurrent : bool, default False
            Accepted for compatibility with Ollama Batch client; ignored here
            because xAI batch is executed remotely by the provider.
        **kwargs
            Forwarded to each per-item request.

        Returns
        -------
        list[xAIResponse] or xAIResponse
            List of responses (one per input conversation) or a single response
            when a single conversation was supplied.
        """
        # Normalise to list-of-conversations for the batch helper
        if (
            isinstance(messages, list)
            and len(messages) > 0
            and isinstance(messages[0], dict)
        ):
            # Single conversation → wrap so batch helper returns list of 1
            messages_list: list[list[dict[str, Any]]] = [messages]
            single_result = True
        else:
            messages_list = cast(list[list[dict[str, Any]]], messages)
            single_result = False

        results = await create_batch_chat(
            self,
            messages_list,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            response_model=response_model,
            **kwargs,
        )

        if single_result:
            return results[0]
        return results


class EmbedXAIClient(BaseXAIClient):
    """Embeddings client for xAI (uses OpenAI-compatible /v1/embeddings endpoint).

    Persists with ``kind="embedding"`` (branching=False) when ``save_mode``
    is enabled. This is a convenience wrapper; for advanced use call the
    generic HTTP client directly.
    """

    async def embeddings(
        self,
        input: str | list[str],
        model: str = "text-embedding-3-large",
        *,
        save_mode: SaveMode = "none",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate embeddings for one or more texts.

        Parameters
        ----------
        input : str or list[str]
            Text(s) to embed.
        model : str, default "text-embedding-3-large"
            Embedding model (xAI currently exposes OpenAI-compatible names).
        save_mode : {"none", "json_files", "postgres"}, default "none"
            Persistence backend.
        **kwargs
            Extra payload fields (e.g. dimensions, encoding_format).

        Returns
        -------
        dict
            Raw embeddings response from xAI (data, model, usage, ...).
        """
        self.logger.info(
            "xAI embeddings request",
            extra={
                "model": model,
                "input_count": len(input) if isinstance(input, list) else 1,
            },
        )

        http_client = await self._get_http_client()
        payload = {"input": input, "model": model, **kwargs}
        try:
            resp = await http_client.post("/v1/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self.logger.error("xAI embeddings failed", extra={"error": str(exc)})
            raise

        # Optional persistence (embedding kind, no branching)
        if save_mode != "none" and self.persistence_manager is not None:
            try:
                # Use the generic persist path; embedding responses are stored
                # under kind="embedding" with branching=False
                await self.persistence_manager.persist_chat_turn(
                    data,                                                                 # provider_response (dict fallback; real impl should wrap in protocol)
                    {"input": input, "model": model},                                     # provider_request
                    kind="embedding",
                    branching=False,
                )
            except Exception as exc:
                self.logger.warning(
                    "Embedding persistence failed (continuing)",
                    extra={"error": str(exc)},
                )

        return data


def XAIClient(
    logger: logging.Logger,
    mode: ChatMode = "turn",
    api_key: str = "",
    persistence_manager: "PersistenceManager | None" = None,
    **kwargs: Any,
) -> TurnXAIClient | StreamXAIClient | BatchXAIClient | EmbedXAIClient:
    """Factory returning the appropriate xAI client for the requested mode.

    This is the function registered with ``client_factory.register_provider``.
    End users normally obtain instances via the unified
    ``get_llm_client("xai", mode=..., ...)`` entry point.

    Parameters
    ----------
    logger : logging.Logger
        Structured logger.
    mode : {"turn", "stream", "batch"}, default "turn"
        Desired interaction mode. "embed" is not supported here (use
        ``EmbedXAIClient`` directly for embeddings).
    api_key : str
        xAI API key.
    persistence_manager : PersistenceManager, optional
        Shared persistence backend.
    **kwargs
        Passed through to the concrete client constructor (timeout, base_url, ...).

    Returns
    -------
    TurnXAIClient | StreamXAIClient | BatchXAIClient | EmbedXAIClient
        Concrete client instance for the chosen mode.
    """
    if mode == "turn":
        return TurnXAIClient(
            logger, api_key=api_key, persistence_manager=persistence_manager, **kwargs
        )
    elif mode == "stream":
        return StreamXAIClient(
            logger, api_key=api_key, persistence_manager=persistence_manager, **kwargs
        )
    elif mode == "batch":
        return BatchXAIClient(
            logger, api_key=api_key, persistence_manager=persistence_manager, **kwargs
        )
    else:
        raise ValueError(f"Unsupported xAI mode: {mode}")


# Auto-register with the central factory (turn / stream / batch only)
from .client_factory import register_provider

register_provider("xai", TurnXAIClient, StreamXAIClient, BatchXAIClient)
