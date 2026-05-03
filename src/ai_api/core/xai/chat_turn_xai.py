"""
xAI Turn-Based (Non-Streaming) Chat Implementation (SDK-based)

This module provides the concrete implementation for single-turn chat
completions against the xAI API using the official xAI SDK. It is the
backend for ``TurnXAIClient.create_chat`` and is designed to be called
from the high-level ``ChatSession`` or the public ``xAIClient``.

Key responsibilities
--------------------
- Build an ``xAIRequest`` (implements ``LLMRequestProtocol``).
- Support structured JSON output via ``xAIJSONResponseSpec``.
- Forward branching metadata (``tree_id``, ``branch_id``,
  ``parent_response_id``, ``sequence``, ``neutral_history_slice``) via
  ``**kwargs`` to ``PersistenceManager.persist_chat_turn``.
- Use the unified ``persist_chat_turn`` (single call for request + response
  + branching) instead of separate persist_request / persist_response.
- Validate structured output with Pydantic ``response_model`` and attach
  ``.parsed``.

Git-style Branching Support
---------------------------
All branching identifiers are passed through ``**kwargs`` and forwarded
to ``persist_chat_turn``. This enables ``ChatSession.create_or_continue``
and ``edit_history`` (rebase) to work transparently with xAI without any
changes to the public client API.

See Also
--------
ai_api.core.xai_client.TurnXAIClient
ai_api.core.common.chat_session.ChatSession
ai_api.core.common.persistence.PersistenceManager.persist_chat_turn
ai_api.core.xai.chat_stream_xai
ai_api.core.xai.chat_batch_xai
ai_api.core.ollama.chat_turn_ollama
"""

from typing import Any, Type

from pydantic import BaseModel

from ...data_structures.xai_objects import xAIRequest, xAIResponse
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
    """Create a single turn-based (non-streaming) chat completion with xAI.

    This is the low-level implementation used by ``TurnXAIClient`` and
    ``ChatSession.create_or_continue``. It builds the request, calls the
    xAI SDK, optionally validates structured output, and persists the
    interaction via the unified ``persist_chat_turn`` (which handles
    request + response + Git-style branching metadata in one atomic step).

    Parameters
    ----------
    client : Any
        The outer ``xAIClient`` (or ``ChatSession`` wrapper) providing
        ``logger`` and ``persistence_manager``.
    sdk_client : Any
        Pre-initialised xAI ``AsyncClient`` (from ``client._sdk_client``).
    messages : list[dict]
        OpenAI-compatible message list (system/user/assistant/tool).
    model : str, default "grok-4"
        xAI model identifier.
    temperature : float or None, optional
        Sampling temperature.
    max_tokens : int or None, optional
        Maximum tokens to generate.
    save_mode : {"none", "json_files", "postgres"}, default "none"
        Persistence backend.
    response_model : type[BaseModel] or None, optional
        Pydantic model for structured output. Converted internally via
        ``create_json_response_spec``.
    **kwargs : Any
        Forwarded to ``xAIRequest`` and to ``persist_chat_turn``.
        Branching keys (``tree_id``, ``branch_id``, ``parent_response_id``,
        ``sequence``, ``neutral_history_slice``) are automatically
        extracted and passed to the persistence layer, enabling
        ``ChatSession`` and ``edit_history`` (rebase) workflows.

    Returns
    -------
    xAIResponse
        Completed response object. If ``response_model`` was supplied,
        the ``.parsed`` attribute contains the validated Pydantic instance.

    Notes
    -----
    This function no longer calls ``persist_request`` + ``persist_response``
    separately. It now uses the single ``persist_chat_turn`` entry point
    so that branching metadata and the new ``Conversations`` table are
    updated atomically.
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

    #  3. Call the xAI SDK (chat completion)
    chat = sdk_client.chat.create(
        model=request.model,
        **request.to_sdk_chat_kwargs(),
    )
    raw_response = await chat.completions.create()
    response = xAIResponse.from_sdk(raw_response)

    # 4. Validate the response if a response specification is provided.
    if response_model is not None:
        try:
            parsed = response_model.model_validate_json(response.text)
            response.parsed = parsed
        except Exception as exc:
            client.logger.warning(
                "Failed to parse structured response", extra={"error": str(exc)}
            )

    # 5. Persist via unified entry point (handles request + response + branching)
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_chat_turn(
                provider_response=response,
                provider_request=request,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    in (
                        "tree_id",
                        "branch_id",
                        "parent_response_id",
                        "sequence",
                        "neutral_history_slice",
                    )
                },
            )
        except Exception as exc:
            logger.warning(
                "Persistence via persist_chat_turn failed (continuing)",
                extra={"error": str(exc)},
            )

    logger.info("Turn-based xAI chat completed", extra={"model": model})
    return response
