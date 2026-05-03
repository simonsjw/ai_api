"""
Ollama embeddings (non-conversational path) with GPU memory availability checks.

This module implements ``create_embeddings`` (called by ``EmbedOllamaClient.create_chat``).
It generates vector embeddings using Ollama's ``/api/embeddings`` (or ``/api/embed``)
endpoint and persists via the unified ``persist_chat_turn`` path with ``kind="embedding"``.

GPU Memory Check
----------------
The embedding POST is wrapped with the same ``is_ollama_gpu_memory_error`` +
``wrap_ollama_gpu_mem_error`` logic used by chat methods. Embedding models
can still trigger VRAM exhaustion on consumer GPUs when the model is large
or many texts are embedded in one call.

See Also
--------
chat_turn_ollama, chat_stream_ollama : same GPU guard pattern.
ai_api.core.ollama.errors_ollama : central detector + wrapper.
"""

from __future__ import annotations

import logging
from typing import Any, Union

import httpx

from ...data_structures.ollama_objects import OllamaRequest
from .errors_ollama import (
    is_ollama_gpu_memory_error,
    wrap_ollama_api_error,
    wrap_ollama_gpu_mem_error,
)


class OllamaEmbedResponse:
    """Lightweight container for Ollama embedding results."""

    def __init__(
        self,
        model: str,
        embeddings: list[Any],
        usage: dict,
        raw: dict,
    ):
        self.model = model
        self.embeddings = embeddings
        self.usage = usage
        self.raw = raw

    def to_neutral_format(self, branch_info: dict | None = None) -> dict[str, Any]:
        return {
            "role": "embedding",
            "content": self.embeddings,
            "structured": None,
            "finish_reason": "stop",
            "usage": self.usage,
            "raw": self.raw,
            "branch_meta": branch_info or {},
        }


async def create_embeddings(
    client: Any,
    input: Union[str, list[str]],
    model: str = "nomic-embed-text",
    *,
    save_mode: str = "none",
    **kwargs: Any,
) -> OllamaEmbedResponse:
    """Generate embeddings for one or more strings (with GPU memory guard).

    Parameters
    ----------
    client : BaseOllamaClient
        Client instance providing logger, persistence_manager and HTTP client.
    input : str or list of str
        Text(s) to embed.
    model : str, default "nomic-embed-text"
        Ollama embedding model.
    save_mode : {"none", "json_files", "postgres"}, default "none"
    **kwargs
        Extra parameters forwarded to the Ollama ``/api/embeddings`` payload.

    Returns
    -------
    OllamaEmbedResponse

    Raises
    ------
    OllamaGPUMemoryWarning
        If the local GPU cannot allocate enough VRAM for the embedding model.
    httpx.HTTPStatusError (wrapped)
        For other API errors.
    """
    logger = client.logger
    logger.info(
        "Creating Ollama embeddings",
        extra={"model": model, "batch": isinstance(input, list)},
    )

    payload: dict[str, Any] = {"model": model, "input": input}
    if "options" in kwargs:
        payload["options"] = kwargs.pop("options")

    http_client = await client._get_http_client()

    try:
        resp = await http_client.post("/api/embeddings", json=payload)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        if is_ollama_gpu_memory_error(exc):
            raise wrap_ollama_gpu_mem_error(
                exc,
                f"GPU memory exhausted while generating embeddings with model '{model}' "
                "(embedding models can still require substantial VRAM on large inputs)",
            ) from exc
        if isinstance(exc, httpx.HTTPStatusError):
            raise wrap_ollama_api_error(
                exc, f"Ollama embeddings request failed for model '{model}'"
            ) from exc
        raise

    embeddings = raw.get("embeddings") or raw.get("embedding") or []
    usage = raw.get("usage", {})

    response = OllamaEmbedResponse(
        model=model, embeddings=embeddings, usage=usage, raw=raw
    )

    # Persist (non-chat, no branching)
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            req = OllamaRequest(model=model, input=input, save_mode=save_mode, **kwargs)
            await client.persistence_manager.persist_chat_turn(
                provider_response=response,
                provider_request=req,
                kind="embedding",
                branching=False,
            )
        except Exception as exc:
            logger.warning(
                "Embedding persistence failed (continuing)", extra={"error": str(exc)}
            )

    logger.info("Ollama embeddings completed", extra={"model": model})
    return response
