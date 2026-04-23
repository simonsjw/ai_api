"""
Ollama turn-based (non-streaming) chat implementation.

This module provides the core logic for single-turn chat completions against
Ollama's native HTTP API (/api/chat). It is the concrete implementation
behind ``TurnOllamaClient.create_chat(...)``.

High-level view of Ollama-specific functionality
------------------------------------------------
- Uses direct httpx POST to the local Ollama server (no SDK).
- Builds an ``OllamaRequest`` (which implements ``LLMRequestProtocol``) from
  the high-level parameters.
- Supports the full set of Ollama generation options via ``**kwargs``
  (temperature, top_p, top_k, num_ctx, repeat_penalty, seed, stop, mirostat,
  think, etc.).
- Structured JSON output is handled by attaching an ``OllamaJSONResponseSpec``
  (created via the common ``create_json_response_spec`` helper or directly).
- Request and response are persisted symmetrically using the protocol methods
  (``meta()``, ``payload()``, ``endpoint()``) so the same persistence layer
  works for both Ollama and xAI.

Comparison with xAI turn-based chat
-----------------------------------
- Ollama: native HTTP, more low-level generation parameters, local execution
  (no network latency, full control over model files).
- xAI: uses the official xAI SDK, fewer generation parameters exposed at this
  level, remote execution, native support for batching (see ``chat_batch_xai``)
  and richer error taxonomy (gRPC, rate limits, thinking mode).

The function is deliberately thin — all heavy lifting (request construction,
persistence, structured-output parsing) is delegated to reusable components
in ``common/`` and ``data_structures/ollama_objects.py``.

See Also
--------
ai_api.core.ollama_client.TurnOllamaClient
    The public client that calls this function.
ai_api.core.ollama.chat_stream_ollama
    The streaming counterpart.
ai_api.core.xai.chat_turn_xai
    The xAI equivalent (SDK-based).
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
from ..common.persistence import PersistenceManager


async def create_turn_chat_session(
    client: Any,  # OllamaClient or similar
    messages: list[dict],
    model: str = "llama3.2",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    save_mode: str = "none",
    response_model: Type[BaseModel] | None = None,
    **kwargs: Any,
) -> OllamaResponse:
    """Create a turn-based (non-streaming) chat session with Ollama.

    This is the workhorse for ``TurnOllamaClient.create_chat``. It constructs
    an ``OllamaRequest``, optionally attaches a structured-output spec,
    calls the native ``/api/chat`` endpoint, and persists both request and
    response using the symmetrical protocol pattern.

    Parameters
    ----------
    client : Any
        An object with ``logger``, ``persistence_manager``, and
        ``_get_http_client()`` (typically ``TurnOllamaClient`` or
        ``BaseOllamaClient``).
    messages : list[dict]
        Chat history in OpenAI-style format (list of {"role": ..., "content": ...}).
    model : str, optional
        Ollama model name (default "llama3.2").
    temperature : float or None, optional
        Sampling temperature.
    max_tokens : int or None, optional
        Maximum tokens to generate.
    save_mode : {"none", "json_files", "postgres"}, optional
        Persistence mode (default "none").
    response_model : type[pydantic.BaseModel] or None, optional
        If provided, a Pydantic model that the response must validate against.
        Internally converted to ``OllamaJSONResponseSpec``.
    **kwargs
        Additional Ollama-specific generation parameters (top_p, top_k, num_ctx,
        repeat_penalty, seed, stop, mirostat, think, etc.). Passed through to
        ``OllamaRequest``.

    Returns
    -------
    OllamaResponse
        The completed response object (with optional ``.parsed`` attribute if
        ``response_model`` was supplied).

    Examples
    --------
    Basic usage (called from client):

    .. code-block:: python

        response = await create_turn_chat_session(
            client,
            messages=[{"role": "user", "content": "Hello!"}],
            model="llama3.2",
            temperature=0.7,
            save_mode="postgres",
        )

    With structured output:

    .. code-block:: python

        class Person(BaseModel):
            name: str
            age: int

        response = await create_turn_chat_session(
            client,
            messages=[{"role": "user", "content": "Extract person info"}],
            model="llama3.2",
            response_model=Person,
        )
        print(response.parsed)  # validated Person instance
    """

    logger = client.logger
    logger.info("Creating turn-based Ollama chat", extra={"model": model})

    # Handle structured output
    response_format: OllamaJSONResponseSpec | None = None
    if response_model is not None:
        response_format = OllamaJSONResponseSpec(model=response_model)
    elif "response_format" in kwargs:
        # Allow direct passing of OllamaJSONResponseSpec or dict via **kwargs
        rf = kwargs.pop("response_format")
        if isinstance(rf, dict):
            response_format = OllamaJSONResponseSpec(model=rf)
        elif isinstance(rf, OllamaJSONResponseSpec):
            response_format = rf
        # else: let it be handled by OllamaRequest validation if possible

    # 1. Build request
    request = OllamaRequest(
        model=model,
        input=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        save_mode=save_mode,
        response_format=response_format,
        **kwargs,
    )

    # 2. Persist request (protocol style)
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_request(request)
        except Exception as exc:
            logger.warning(
                "Request persistence failed (continuing)", extra={"error": str(exc)}
            )

    # 3. Call Ollama
    http_client = await client._get_http_client()
    payload = request.to_ollama_dict()
    payload["stream"] = False

    resp = await http_client.post("/api/chat", json=payload)
    resp.raise_for_status()
    raw_data = resp.json()

    response = OllamaResponse.from_dict(raw_data)

    # 4. Persist response (NEW symmetrical protocol style)
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_response(response, request=request)
        except Exception as exc:
            logger.warning(
                "Response persistence failed (continuing)", extra={"error": str(exc)}
            )

    logger.info("Turn-based Ollama chat completed", extra={"model": model})
    return response
