"""
Ollama embeddings implementation (native /api/embed endpoint).

This module provides a complete, self-contained embeddings solution for Ollama.
It defines its own request and response models that implement the
``LLMRequestProtocol`` and ``LLMResponseProtocol`` so they integrate
seamlessly with the shared persistence layer.

High-level view of Ollama embeddings
------------------------------------
- Calls the dedicated ``/api/embed`` endpoint (not the chat endpoint).
- Supports single string or list of strings as input.
- Returns rich telemetry: total_duration, load_duration, prompt_eval_count,
  embedding dimension, etc.
- Full support for ``save_mode`` (json_files or postgres) via the protocol
  methods.
- Exposes a thin ``EmbedOllamaClient`` wrapper for convenience.

Ollama vs xAI embeddings
------------------------
- Ollama: native local endpoint, detailed performance telemetry, supports
  ``dimensions`` and ``truncate`` parameters, no remote latency.
- xAI: does not expose a dedicated embeddings module in this codebase
  (embeddings would be handled via the chat endpoint or a future xAI
  embeddings client). Ollama is currently the only provider with a
  first-class embeddings implementation.

The request/response dataclasses are frozen Pydantic models for immutability
and type safety.

See Also
--------
ai_api.core.ollama_client.EmbedOllamaClient
    Public client that delegates to ``create_embeddings``.
ai_api.data_structures.base_objects
    The protocols implemented by ``OllamaEmbedRequest`` / ``OllamaEmbedResponse``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx
from pydantic import BaseModel, ConfigDict

from ...data_structures.base_objects import (
    LLMEndpoint,
    LLMRequestProtocol,
    LLMResponseProtocol,
)
from ..common.persistence import PersistenceManager
from .errors_ollama import wrap_ollama_error

__all__ = [
    "create_embeddings",
    "EmbedOllamaClient",
    "OllamaEmbedRequest",
    "OllamaEmbedResponse",
]


# ----------------------------------------------------------------------
# Self-contained Request model (implements LLMRequestProtocol)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaEmbedRequest(BaseModel):
    """Request model for Ollama embeddings.

    Implements ``LLMRequestProtocol`` so it can be persisted identically
    to chat requests.

    Parameters
    ----------
    model : str
        Ollama model name (must support embeddings, e.g. "nomic-embed-text").
    input : str or Sequence[str]
        Single text or list of texts to embed.
    truncate : bool, optional
        Whether to truncate inputs that exceed context (default True).
    options : dict or None, optional
        Additional model options.
    keep_alive : str, int or None, optional
        How long to keep the model loaded (e.g. "5m", 300).
    dimensions : int or None, optional
        Desired embedding dimensionality (if supported by model).
    """

    model: str
    input: str | Sequence[str]
    truncate: bool = True
    options: dict[str, Any] | None = None
    keep_alive: str | int | None = None
    dimensions: int | None = None

    model_config = ConfigDict(frozen=True)

    def meta(self) -> dict[str, Any]:
        """Return generation / embedding settings for persistence."""
        return {
            "truncate": self.truncate,
            "dimensions": self.dimensions,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }

    def payload(self) -> dict[str, Any]:
        """Return the actual input texts for persistence."""
        inp = self.input if isinstance(self.input, str) else list(self.input)
        return {
            "input": inp,
            "input_type": "embeddings",
            "n_inputs": 1 if isinstance(self.input, str) else len(inp),
        }

    def endpoint(self) -> LLMEndpoint:
        """Return structured endpoint information."""
        return LLMEndpoint(
            provider="ollama",
            model=self.model,
            base_url="http://localhost:11434",
            path="/api/embed",
            api_type="native",
        )

    def to_ollama_dict(self) -> dict[str, Any]:
        """Convert to the exact payload expected by /api/embed."""
        return {
            "model": self.model,
            "input": self.input if isinstance(self.input, str) else list(self.input),
            "truncate": self.truncate,
            "keep_alive": self.keep_alive,
            "dimensions": self.dimensions,
        }


# ----------------------------------------------------------------------
# Self-contained Response model (implements LLMResponseProtocol)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaEmbedResponse(BaseModel):
    """Response model for Ollama embeddings.

    Implements ``LLMResponseProtocol``. Contains rich telemetry not
    usually exposed by other providers.

    Attributes
    ----------
    model : str
    embeddings : list[list[float]]
        The actual embedding vectors.
    total_duration, load_duration, prompt_eval_count, prompt_eval_duration : int or None
        Ollama-native performance counters (nanoseconds).
    raw : dict
        The complete raw response from Ollama.
    """

    model: str
    embeddings: list[list[float]]
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    @property
    def n_inputs(self) -> int:
        """Number of input texts that were embedded."""
        return len(self.embeddings)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of each embedding vector."""
        return len(self.embeddings[0]) if self.embeddings else 0

    def meta(self) -> dict[str, Any]:
        """Return metadata for persistence."""
        return {
            "model": self.model,
            "n_inputs": self.n_inputs,
            "embedding_dim": self.embedding_dim,
            "total_duration": self.total_duration,
            "load_duration": self.load_duration,
        }

    def payload(self) -> dict[str, Any]:
        """Return the embeddings and derived telemetry for persistence."""
        return {
            "embeddings": self.embeddings,
            "n_inputs": self.n_inputs,
            "embedding_dim": self.embedding_dim,
            "telemetry": {
                "total_duration_ms": round(self.total_duration / 1_000_000, 2)
                if self.total_duration
                else None,
            },
        }

    def endpoint(self) -> LLMEndpoint:
        """Return structured endpoint information (same as request)."""
        return LLMEndpoint(
            provider="ollama",
            model=self.model,
            base_url="http://localhost:11434",
            path="/api/embed",
            api_type="native",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OllamaEmbedResponse":
        """Construct from the raw JSON returned by /api/embed."""
        return cls(
            model=data["model"],
            embeddings=data.get("embeddings", []),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            prompt_eval_duration=data.get("prompt_eval_duration"),
            raw=data,
        )


# ----------------------------------------------------------------------
# Core embedding logic
# ----------------------------------------------------------------------


async def create_embeddings(
    client: "EmbedOllamaClient",
    model: str,
    input: str | Sequence[str],
    *,
    truncate: bool = True,
    options: dict[str, Any] | None = None,
    keep_alive: str | int | None = None,
    dimensions: int | None = None,
    save_mode: str = "none",
    **kwargs: Any,
) -> OllamaEmbedResponse:
    """Generate embeddings using Ollama's native /api/embed endpoint.

    This is the function called by ``EmbedOllamaClient.create_embeddings``.

    Parameters
    ----------
    client : EmbedOllamaClient
        Client instance providing logger, HTTP client, and persistence manager.
    model : str
        Ollama embedding model.
    input : str or Sequence[str]
        Text(s) to embed.
    truncate, options, keep_alive, dimensions, save_mode, **kwargs
        Forwarded to ``OllamaEmbedRequest``.

    Returns
    -------
    OllamaEmbedResponse
        Response containing the embeddings and rich telemetry.
    """

    client.logger.info(
        "Creating Ollama embeddings",
        extra={"model": model, "n_inputs": 1 if isinstance(input, str) else len(input)},
    )

    request = OllamaEmbedRequest(
        model=model,
        input=input,
        truncate=truncate,
        options=options,
        keep_alive=keep_alive,
        dimensions=dimensions,
    )

    # Persist request
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_request(request)
        except Exception as exc:
            client.logger.warning(
                "Request persistence failed (continuing)", extra={"error": str(exc)}
            )

    http_client = await client._get_http_client()
    payload = request.to_ollama_dict()

    try:
        resp = await http_client.post("/api/embed", json=payload)
        resp.raise_for_status()
        raw_data = resp.json()
    except Exception as exc:
        client.logger.error(
            "Ollama embeddings request failed", extra={"error": str(exc)}
        )
        raise wrap_ollama_error(
            exc, "Failed to generate embeddings via Ollama"
        ) from exc

    response = OllamaEmbedResponse.from_dict(raw_data)

    # Persist response (symmetrical protocol style)
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_response(response, request=request)
        except Exception as exc:
            client.logger.warning(
                "Response persistence failed (continuing)", extra={"error": str(exc)}
            )

    client.logger.info(
        "Ollama embeddings completed",
        extra={
            "model": model,
            "n_vectors": response.n_inputs,
            "dim": response.embedding_dim,
        },
    )
    return response


class EmbedOllamaClient:
    """Dedicated embeddings client for Ollama.

    Provides the same lifecycle (``_get_http_client``, ``aclose``) as the
    chat clients and a thin ``create_embeddings`` wrapper.

    Parameters
    ----------
    logger : logging.Logger
    host : str, optional
        Ollama base URL (default "http://localhost:11434").
    timeout : int or None, optional
        Request timeout in seconds.
    persistence_manager : PersistenceManager or None, optional
        Shared persistence instance.
    """

    def __init__(
        self,
        logger: logging.Logger,
        host: str = "http://localhost:11434",
        timeout: int | None = 180,
        persistence_manager: PersistenceManager | None = None,
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

    async def create_embeddings(self, *args, **kwargs) -> OllamaEmbedResponse:
        """Convenience wrapper around the module-level ``create_embeddings``."""
        return await create_embeddings(self, *args, **kwargs)

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
