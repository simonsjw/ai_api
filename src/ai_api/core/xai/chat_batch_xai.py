"""
xAI batch chat processing with flexible structured output.

This module is **unique to xAI** — Ollama has no native batch endpoint, so
batch support for Ollama is simulated via concurrent/sequential turn calls
in ``ollama_client.py``.

High-level view of xAI batch functionality
------------------------------------------
- Accepts a list of message lists (one conversation per batch item).
- Supports three modes for ``response_model``:
  - ``None`` → normal unstructured batch
  - Single ``Type[BaseModel]`` → same model applied to every request
  - ``list[Type[BaseModel]]`` → different model per request (length must
    exactly match batch size)
- Re-uses the turn-based logic (``create_turn_chat_session``) for each item
  to keep the code DRY and ensure consistent persistence + structured-output
  handling.
- Validates list length when per-request models are supplied.

Comparison with Ollama
----------------------
- xAI: true native batch endpoint via the SDK, per-request structured output
  flexibility, efficient remote execution.
- Ollama: no native batch; the client simulates it with ``asyncio.gather``
  (concurrent) or sequential execution, with GPU-memory warnings. No
  per-request ``response_model`` list support at the batch level.

This module demonstrates the power of the shared ``create_json_response_spec``
helper and the symmetrical persistence pattern — the same code paths are
used whether you call turn, stream, or batch.

See Also
--------
ai_api.core.xai_client (the public client exposing batch mode)
ai_api.core.xai.chat_turn_xai
    The per-item implementation reused here.
ai_api.core.ollama_client.BatchOllamaClient
    The simulated batch implementation for Ollama.
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from ...data_structures.xai_objects import xAIRequest
from ..common.response_struct import create_json_response_spec


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
    Process a batch of conversations with optional per-request structured output.

    Parameters
    ----------
    client : Any
        xAIClient providing logger, persistence_manager, and ``_sdk_client``.
    messages_list : list[list[dict]]
        List of conversations (each is a list of message dicts).
    model : str
        xAI model to use for all requests in the batch.
    temperature, max_tokens, save_mode, **kwargs
        Common generation parameters applied to every request.
    response_model : Type[BaseModel], list[Type[BaseModel]], or None, optional
        - None: no structured output
        - Single model: same model for every item
        - List of models: one model per item (must match ``len(messages_list)``)

    Returns
    -------
    list[Any]
        List of ``xAIResponse`` objects (one per input conversation).

    Raises
    ------
    ValueError
        If ``response_model`` is a list whose length does not match the batch size.
    """

    logger = client.logger
    logger.info("Starting xAI batch chat", extra={"batch_size": len(messages_list)})

    # === VALIDATION: if list, must match batch size ===
    if isinstance(response_model, list):
        if len(response_model) != len(messages_list):
            raise ValueError(
                f"response_model list length ({len(response_model)}) "
                f"must equal number of batch requests ({len(messages_list)})"
            )

    results = []

    for idx, messages in enumerate(messages_list):
        # Pick the right response_model for this specific request
        if isinstance(response_model, list):
            current_rm = response_model[idx]
        else:
            current_rm = response_model  # None or single shared model

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
            spec = create_json_response_spec("xai", current_rm)
            request = request.model_copy(
                update={"response_format": spec.to_sdk_response_format()}
            )

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
