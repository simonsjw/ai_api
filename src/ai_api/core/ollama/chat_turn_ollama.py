"""
Ollama turn-based (non-streaming) chat with Git-style branching support
and GPU memory availability checks.

This module provides the coroutine ``create_turn_chat_session`` used by
``TurnOllamaClient.create_chat`` (and the batch simulator). It executes a
single HTTP POST to Ollama's ``/api/chat`` endpoint, converts the response
into an ``OllamaResponse``, and persists the interaction **exactly once**
via the unified ``PersistenceManager.persist_chat_turn`` entry point.

GPU Memory Check
----------------
The HTTP call is guarded by ``is_ollama_gpu_memory_error`` + ``wrap_ollama_gpu_mem_error``.
If Ollama returns an error indicating VRAM exhaustion, a typed
``OllamaGPUMemoryWarning`` is raised so callers can react (smaller context,
different model, CPU fallback, etc.).

See Also
--------
chat_stream_ollama : streaming counterpart with identical GPU guard.
ai_api.core.ollama.errors_ollama : detector + wrapper implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Type

import httpx
from pydantic import BaseModel

from ...data_structures.ollama_objects import (
    OllamaJSONResponseSpec,
    OllamaRequest,
    OllamaResponse,
)
from .errors_ollama import (
    OllamaClientError,
    is_ollama_gpu_memory_error,
    wrap_ollama_api_error,
    wrap_ollama_gpu_mem_error,
)


async def create_turn_chat_session(
    client: Any,
    messages: list[dict[str, Any]],
    model: str = "llama3.2",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    save_mode: str = "none",
    response_model: Type[BaseModel] | None = None,
    strict_structured_output: bool = False,
    **kwargs: Any,
) -> OllamaResponse:
    """Execute a single non-streaming chat turn against Ollama (with GPU memory guard).

    Parameters
    ----------
    client : BaseOllamaClient
        The Ollama client instance (provides logger, persistence_manager, _get_http_client).
    messages : list of dict
        Conversation history in OpenAI-style format.
    model : str, default "llama3.2"
        Ollama model name.
    temperature, max_tokens : optional
        Generation parameters.
    save_mode : {"none", "json_files", "postgres"}, default "none"
    response_model : type[BaseModel], optional
        Pydantic model for structured JSON output.
    **kwargs
        Forwarded to OllamaRequest and persist_chat_turn (including branching metadata).

    Returns
    -------
    OllamaResponse

    Raises
    ------
    OllamaGPUMemoryWarning
        If local GPU runs out of VRAM for this model + context.
    httpx.HTTPStatusError (wrapped)
        For other Ollama API errors.
    """
    logger = client.logger
    logger.info("Creating turn-based Ollama chat", extra={"model": model})

    # Structured output handling
    response_format: OllamaJSONResponseSpec | None = None
    if response_model is not None:
        response_format = OllamaJSONResponseSpec(model=response_model)
    elif "response_format" in kwargs:
        rf = kwargs.pop("response_format")
        if isinstance(rf, dict):
            response_format = OllamaJSONResponseSpec(model=rf)
        elif isinstance(rf, OllamaJSONResponseSpec):
            response_format = rf

    request = OllamaRequest(
        model=model,
        input=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        save_mode=save_mode,
        response_format=response_format,
        **kwargs,
    )

    http_client = await client._get_http_client()
    payload = request.to_ollama_dict()
    payload["stream"] = False

    try:
        resp = await http_client.post("/api/chat", json=payload)
        resp.raise_for_status()
        raw_data = resp.json()
    except Exception as exc:
        if is_ollama_gpu_memory_error(exc):
            raise wrap_ollama_gpu_mem_error(
                exc,
                f"GPU memory exhausted while running chat with model '{model}' "
                "(try reducing num_ctx or using a smaller model)",
            ) from exc
        if isinstance(exc, httpx.HTTPStatusError):
            raise wrap_ollama_api_error(
                exc, f"Ollama chat request failed for model '{model}'"
            ) from exc
        raise

    response = OllamaResponse.from_dict(raw_data)

    # Structured output parsing (if response_model supplied)
    if response_model is not None and response.text:
        try:
            parsed = response_model.model_validate_json(response.text)
            response.parsed = parsed
        except Exception as exc:
            if strict_structured_output:
                raise OllamaClientError(
                    f"Failed to parse structured response for model '{model}' "
                    "(strict mode enabled)",
                    details={"original": type(exc).__name__},
                ) from exc
            else:
                logger.warning(
                    "Failed to parse structured response – continuing with raw text",
                    extra={
                        "error": str(exc),
                        "model": model,
                        "response_text_preview": (response.text or "")[:200],
                        "error_type": type(exc).__name__,
                    },
                )
                # best-effort: return raw response (parsed left unset)

    # Persistence
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_chat_turn(
                provider_response=response,
                provider_request=request,
                kind="chat",
                branching=True,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("tree_id", "branch_id", "parent_response_id", "sequence")
                },
            )
        except Exception as exc:
            logger.warning(
                "Persistence via persist_chat_turn failed (continuing)",
                extra={"error": str(exc)},
            )

    logger.info("Turn-based Ollama chat completed", extra={"model": model})
    return response
