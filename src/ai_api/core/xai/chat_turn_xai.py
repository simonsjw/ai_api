"""
xAI turn-based (non-streaming) chat implementation (SDK-based).

This module provides the core logic for single-turn chat completions against
the xAI API using the official xAI SDK. It is the concrete implementation
behind the turn-based path of ``xAIClient``.

High-level view of xAI turn-based functionality
-----------------------------------------------
- Receives a pre-configured xAI SDK client (``sdk_client``) from the outer
  ``xAIClient``.
- Builds an ``xAIRequest`` (which implements ``LLMRequestProtocol``).
- Supports structured JSON output via ``xAIJSONResponseSpec`` (created by
  the common helper).
- Uses symmetrical persistence (persist request before call, persist final
  response after).
- After the SDK call, optionally validates the response text against a
  Pydantic ``response_model`` and attaches the parsed instance.

Comparison with Ollama turn-based chat
--------------------------------------
- xAI: remote SDK call, structured output attached via ``response_format``
  on the SDK request, richer remote-specific error handling (see
  ``errors_xai.py``).
- Ollama: direct HTTP to localhost, more generation parameters exposed
  (num_ctx, repeat_penalty, think, etc.), local execution, native
  embeddings support.
- Both now share the exact same persistence and structured-output patterns
  for consistency.

The function is intentionally thin — request construction, persistence, and
structured-output parsing are delegated to reusable components.

See Also
--------
ai_api.core.xai_client (the public client)
ai_api.core.xai.chat_stream_xai
    The streaming counterpart.
ai_api.core.xai.chat_batch_xai
    xAI-specific batch support (not present in Ollama).
ai_api.core.ollama.chat_turn_ollama
    The Ollama equivalent (native HTTP).
"""

import logging
from typing import Any, Type

from pydantic import BaseModel

from ...data_structures.xai_objects import xAIRequest, xAIResponse
from ..common.persistence import PersistenceManager
from ..common.response_struct import create_json_response_spec

__all__: list[str] = ["create_turn_chat_session"]


async def create_turn_chat_session(
    client: Any,
    sdk_client: Any,
    messages: list[dict],
    model: str = "grok-4",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    save_mode: str = "none",
    response_model: Type[BaseModel] | None = None,
    **kwargs: Any,
) -> xAIResponse:
    """Create a turn-based (non-streaming) chat session with xAI via the SDK.

    Parameters
    ----------
    client : Any
        Outer client providing ``logger`` and ``persistence_manager``.
    sdk_client : Any
        The xAI SDK chat client (usually ``client._sdk_client``).
    messages : list[dict]
        Chat history in OpenAI-compatible format.
    model : str, optional
        xAI model name (default "grok-4").
    temperature, max_tokens, save_mode, **kwargs
        Generation parameters forwarded to ``xAIRequest``.
    response_model : type[pydantic.BaseModel] or None, optional
        Pydantic model for structured output. Converted to
        ``xAIJSONResponseSpec`` internally.

    Returns
    -------
    xAIResponse
        Completed response (with optional ``.parsed`` attribute).
    """

    logger = client.logger
    logger.info("Creating turn-based xAI chat", extra={"model": model})

    # 1. Build request
    request = xAIRequest(
        model=model,
        input=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        save_mode=save_mode,
        **kwargs,
    )

    # 2. Include specification for response if provided.
    if response_model is not None:
        spec = create_json_response_spec("xai", response_model)
        # Attach to the request (adjust field name to match your xAIRequest)
        request = request.model_copy(
            update={"response_format": spec.to_sdk_response_format()}
        )

    # 3. Persist request (protocol style)
    request_id = request_tstamp = None
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            (
                request_id,
                request_tstamp,
            ) = await client.persistence_manager.persist_request(request)
        except Exception as exc:
            logger.warning(
                "Request persistence failed (continuing)", extra={"error": str(exc)}
            )

    # 4. Call the actual LLM
    # (simplified - in real code this would use the xAI SDK)
    chat = sdk_client.chat.create(
        model=request.model,
        **request.to_sdk_chat_kwargs(),
    )
    raw_response = await chat.completions.create()
    response = xAIResponse.from_sdk(raw_response)

    # 5. Validate the response if a response specification is provided.
    if response_model is not None:
        try:
            parsed = response_model.model_validate_json(response.text)
            response.parsed = parsed
        except Exception as exc:
            client.logger.warning(
                "Failed to parse structured response", extra={"error": str(exc)}
            )

    # 6. Persist response (NEW symmetrical protocol style)
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_response(response, request=request)
        except Exception as exc:
            logger.warning(
                "Response persistence failed (continuing)", extra={"error": str(exc)}
            )

    logger.info("Turn-based xAI chat completed", extra={"model": model})
    return response
