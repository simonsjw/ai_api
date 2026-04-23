"""
chat_batch_xai.py — supports flexible structured output for batches.

response_model can be:
- None                    → normal unstructured batch
- Single Type[BaseModel]  → same model for every request
- list[Type[BaseModel]]   → different model per request (length must match batch size)
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
    Batch processing with optional per-request or shared structured output.
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
