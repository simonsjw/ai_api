"""
Embeddings handler for Ollama native API (/api/embed).

Mirrors the style and patterns of chat_turn_ollama.py and chat_stream_ollama.py.
Fully supports batch embedding, all Ollama parameters, rich telemetry,
optional persistence, and NumPy/SciPy vector helpers.

This module also implements the generic LLMRequestProtocol and LLMResponseProtocol
so that embeddings can be persisted using the exact same persistence layer as chat.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import httpx
from pydantic import BaseModel

from ...data_structures.base_objects import (
    LLMEndpoint,
    LLMRequestProtocol,
    LLMResponseProtocol,
)
from ...data_structures.ollama_objects import (
    OllamaEmbedRequest as _OllamaEmbedRequest,                                            # type: ignore
)
from ...data_structures.ollama_objects import (
    OllamaEmbedResponse as _OllamaEmbedResponse,                                          # type: ignore
)
from ...data_structures.ollama_objects import (
    SaveMode,
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
# Request class with protocol implementation (lightweight wrapper)
# ----------------------------------------------------------------------


class OllamaEmbedRequest(_OllamaEmbedRequest):
    """Ollama embedding request that also satisfies LLMRequestProtocol."""

    def meta(self) -> dict[str, Any]:
        return {
            "truncate": self.truncate,
            "dimensions": self.dimensions,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "input": self.input if isinstance(self.input, str) else list(self.input),
            "input_type": "embeddings",
            "n_inputs": 1 if isinstance(self.input, str) else len(self.input),
        }

    def endpoint(self) -> LLMEndpoint:
        return LLMEndpoint(
            provider="ollama",
            model=self.model,
            base_url="http://localhost:11434",
            path="/api/embed",
            api_type="native",
        )

    # ----------------------------------------------------------------------
    # Response class with protocol implementation
    # ----------------------------------------------------------------------


class OllamaEmbedResponse(_OllamaEmbedResponse):
    """Ollama embedding response that also satisfies LLMResponseProtocol."""

    def meta(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_inputs": self.n_inputs,
            "embedding_dim": self.embedding_dim,
            "total_duration": self.total_duration,
            "load_duration": self.load_duration,
        }

    def payload(self) -> dict[str, Any]:
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
        return LLMEndpoint(
            provider="ollama",
            model=self.model,
            base_url="http://localhost:11434",
            path="/api/embed",
            api_type="native",
        )

    # ----------------------------------------------------------------------
    # Core embedding creation logic
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
    save_mode: SaveMode = "none",
    **kwargs: Any,
) -> OllamaEmbedResponse:
    """Generate embeddings using Ollama's native /api/embed endpoint.

    The returned OllamaEmbedResponse implements LLMResponseProtocol, so it
    can be persisted with the exact same code path as chat responses.
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

    # Optional request persistence (same pattern as chat)
    request_id = request_tstamp = None
    if save_mode == "postgres" and client.persistence_manager is not None:
        try:
            (
                request_id,
                request_tstamp,
            ) = await client.persistence_manager.persist_request(request)
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

    # Optional response persistence
    if (
        save_mode == "postgres"
        and client.persistence_manager is not None
        and request_id is not None
        and request_tstamp is not None
    ):
        try:
            await client.persistence_manager.persist_response(
                request_id=request_id,
                request_tstamp=request_tstamp,
                api_result={"embeddings": response.embeddings, "model": model},
                request=request,
            )
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
    """Dedicated client for Ollama embeddings.

    The create_embeddings method returns an OllamaEmbedResponse that fully
    implements LLMResponseProtocol, enabling unified persistence across chat + embeddings.
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

    async def create_embeddings(
        self,
        model: str,
        input: str | Sequence[str],
        *,
        truncate: bool = True,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
        dimensions: int | None = None,
        save_mode: SaveMode = "none",
        **kwargs: Any,
    ) -> OllamaEmbedResponse:
        """See module docstring for full documentation."""
        return await create_embeddings(
            self,
            model=model,
            input=input,
            truncate=truncate,
            options=options,
            keep_alive=keep_alive,
            dimensions=dimensions,
            save_mode=save_mode,
            **kwargs,
        )

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
