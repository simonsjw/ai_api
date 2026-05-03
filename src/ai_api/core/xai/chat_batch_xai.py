"""
xAI Batch Chat Processing with Flexible Structured Output

This module implements native batch processing for xAI (unlike Ollama,
which has no native batch endpoint). It is the backend for
``BatchXAIClient.create_chat`` and is designed to be called from
``ChatSession`` when ``mode="batch"``.

Features
--------
- Accepts a list of message lists (one conversation per batch item).
- Supports three ``response_model`` modes:
  - ``None`` → plain text batch
  - Single ``Type[BaseModel]`` → same model for every item
  - ``list[Type[BaseModel]]`` → different model per item (length must match)
- Re-uses ``create_turn_chat_session`` for each item → consistent
  persistence, structured-output handling, and branching support.
- Branching metadata can be supplied per-item via ``**kwargs`` (rare for
  batch but supported for advanced ``ChatSession`` workflows).

Error Handling
--------------
- Mismatched ``response_model`` list length raises ``xAIClientError``
  (via ``wrap_xai_client_error``) for clear client-side validation.
- Structured-output spec creation errors are wrapped as ``xAIClientError``.
- Per-item errors from ``create_turn_chat_session`` (``xAIAPIError``,
  ``xAIClientError``, etc.) propagate naturally — the batch fails fast on
  the first failing item (consistent with most batch semantics).
- All raised exceptions inherit from ``AIAPIError`` (directly or indirectly),
  enabling uniform ``except AIAPIError`` handling at higher layers.

See Also
--------
ai_api.core.xai_client.BatchXAIClient
ai_api.core.common.chat_session.ChatSession
ai_api.core.xai.chat_turn_xai.create_turn_chat_session
ai_api.core.ollama_client.BatchOllamaClient (simulated batch)
"""

from typing import Any, Type

from pydantic import BaseModel

from ...data_structures.xai_objects import xAIRequest
from ..common.errors import wrap_client_error
from ..common.response_struct import create_json_response_spec

__all__: list[str] = ["create_batch_chat"]


async def create_batch_chat(
    client: Any,
    messages_list: list[list[dict]],
    model: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    save_mode: str = "none",
    response_model: list[Type[BaseModel]] | Type[BaseModel] | None = None,
    **kwargs: Any,
) -> list[Any]:
    """
    Process a batch of independent conversations (native xAI batch).

    Each item is processed via ``create_turn_chat_session`` so that
    persistence, structured output, and (optionally) branching metadata
    are handled consistently with single-turn and streaming paths.

    Parameters
    ----------
    client : Any
        Outer ``xAIClient`` (provides ``logger``, ``persistence_manager``,
        ``_sdk_client``).
    messages_list : list[list[dict]]
        Batch of conversations. Each inner list is one conversation's
        message history.
    model : str
        xAI model used for every item in the batch.
    temperature : float or None, optional
        Sampling temperature (applied uniformly).
    max_tokens : int or None, optional
        Max tokens per response.
    save_mode : {"none", "json_files", "postgres"}, default "none"
        Persistence backend (passed to each item).
    response_model : Type[BaseModel], list[Type[BaseModel]], or None, optional
        Structured-output model(s). If a list, length must equal
        ``len(messages_list)``.
    **kwargs : Any
        Forwarded to every ``create_turn_chat_session`` call. Supports
        per-item branching keys when advanced ``ChatSession`` workflows
        are used with batch mode.

    Returns
    -------
    list[xAIResponse]
        One completed response per input conversation.

    Raises
    ------
    xAIClientError
        If ``response_model`` is a list and its length does not equal
        ``len(messages_list)``, or if structured-output spec creation fails.
    xAIAPIError
        Any per-item transport, authentication, or model error from the
        underlying ``create_turn_chat_session`` call (propagates immediately).
    """

    logger = client.logger
    logger.info("Starting xAI batch chat", extra={"batch_size": len(messages_list)})

    # === VALIDATION: if list, must match batch size ===
    if isinstance(response_model, list):
        if len(response_model) != len(messages_list):
            exc = ValueError(
                f"response_model list length ({len(response_model)}) "
                f"must equal number of batch requests ({len(messages_list)})"
            )
            raise wrap_client_error(
                exc,
                "response_model list length mismatch in xAI batch request",
            ) from exc

    results = []

    for idx, messages in enumerate(messages_list):
        # Pick the right response_model for this specific request
        if isinstance(response_model, list):
            current_rm = response_model[idx]
        else:
            current_rm = response_model                                                   # None or single shared model

        # Build request
        request = xAIRequest(
            model=model,
            input=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            **kwargs,
        )

        # Apply structured output spec if needed
        if current_rm is not None:
            try:
                spec = create_json_response_spec("xai", current_rm)
                request = request.model_copy(
                    update={"response_format": spec.to_sdk_response_format()}
                )
            except Exception as exc:
                raise wrap_client_error(
                    exc,
                    f"Failed to build structured-output spec for batch item {idx}",
                ) from exc

        # Reuse the turn-based logic (keeps code DRY)
        from .chat_turn_xai import create_turn_chat_session

        result = await create_turn_chat_session(
            client,
            client._sdk_client,
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            response_model=current_rm,
            **kwargs,
        )
        results.append(result)

    return results
