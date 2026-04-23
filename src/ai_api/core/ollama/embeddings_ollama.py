"""
Embeddings handler for Ollama native API (/api/embed).

Self-contained implementation that defines its own request/response models
so there are no missing attribute errors. Implements the LLM*Protocol for
seamless integration with the persistence layer.
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
from ..ollama_client import BaseOllamaClient
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
    """Request model for Ollama embeddings."""

    model: str
    input: str | Sequence[str]
    truncate: bool = True
    options: dict[str, Any] | None = None
    keep_alive: str | int | None = None
    dimensions: int | None = None

    model_config = ConfigDict(frozen=True)

    def meta(self) -> dict[str, Any]:
        return {
            "truncate": self.truncate,
            "dimensions": self.dimensions,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }

    def payload(self) -> dict[str, Any]:
        inp = self.input if isinstance(self.input, str) else list(self.input)
        return {
            "input": inp,
            "input_type": "embeddings",
            "n_inputs": 1 if isinstance(self.input, str) else len(inp),
        }

    def endpoint(self) -> LLMEndpoint:
        return LLMEndpoint(
            provider="ollama",
            model=self.model,
            base_url="http://localhost:11434",
            path="/api/embed",
            api_type="native",
        )

    def to_ollama_dict(self) -> dict[str, Any]:
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
    """Response model for Ollama embeddings."""

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
        return len(self.embeddings)

    @property
    def embedding_dim(self) -> int:
        return len(self.embeddings[0]) if self.embeddings else 0

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OllamaEmbedResponse":
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
    """Generate embeddings using Ollama's native /api/embed endpoint."""

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


class EmbedOllamaClient(BaseOllamaClient):
    """Dedicated embeddings client (canonical implementation).

    Now properly inherits from BaseOllamaClient so it gets shared
    HTTP lifecycle, logging, persistence, and aclose() for free.
    """

    async def create_embeddings(self, *args, **kwargs) -> OllamaEmbedResponse:
        return await create_embeddings(self, *args, **kwargs)
