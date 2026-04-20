"""Embeddings handler for Ollama native API (/api/embed).

Mirrors the style and patterns of chat_turn_ollama.py and chat_stream_ollama.py.
Fully supports batch embedding, all Ollama parameters, rich telemetry,
optional persistence, and NumPy/SciPy vector helpers.

Usage:
    client = EmbedOllamaClient(logger=logger, host="http://localhost:11434")
    resp = await client.create_embeddings(
        model="nomic-embed-text",
        input=["Hello world", "Ollama embeddings are great"],
    )
    vectors = resp.to_numpy()          # shape (2, 768)
    sim = resp.cosine_similarity(0, 1)
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import httpx
from pydantic import BaseModel

from ...data_structures.ollama_objects import (
    OllamaEmbedRequest,
    OllamaEmbedResponse,
    SaveMode,
)
from ..common.persistence import PersistenceManager
from ..ollama_client import EmbedOllamaClient
from .errors_ollama import wrap_ollama_error

__all__ = ["create_embeddings", "EmbedOllamaClient"]


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

    Parameters
    ----------
    model : str
        Embedding model name.
    input : str | Sequence[str]
        Text(s) to embed. Batch supported.
    truncate, options, keep_alive, dimensions
        See OllamaEmbedRequest for full documentation.
    save_mode : SaveMode
        Optional persistence (reuses your existing PersistenceManager).
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
