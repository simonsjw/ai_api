"""
Generic structured response support for any LLM provider.

This module provides a unified way to request structured (Pydantic-validated)
JSON output from any provider, while allowing provider-specific implementation
details to live in the per-provider chat modules.

Design philosophy:
- Keep the public API simple: pass `response_model=YourModel` to create_chat()
- Provider-specific logic (JSON schema formatting, parsing, streaming final-chunk attachment)
  lives in chat_turn_*.py / chat_stream_*.py
- This file contains only the shared helpers and the public convenience functions.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Type

from pydantic import BaseModel

from ...data_structures.base_objects import (
    LLMResponseProtocol,
    LLMStreamingChunkProtocol,
)


def create_json_response_spec(
    provider: str,
    model: type[BaseModel] | dict[str, Any],
    instruction: str | None = None,                                                       # ← Optional, no automatic default
) -> Any:
    """
    Returns the appropriate JSON response spec for the given provider.

    The `instruction` parameter is kept for advanced/power-user cases but
    is completely optional. Modern providers (xAI + Ollama) enforce schemas
    natively, so extra instructions are rarely needed.

    Args:
        provider: "xai" or "ollama"
        model: Pydantic model class or raw JSON schema dict
        instruction: Optional extra guidance (rarely needed in 2026)
    """
    if provider == "xai":
        from ...data_structures.xai_objects import xAIJSONResponseSpec

        return xAIJSONResponseSpec(model=model, instruction=instruction)
    elif provider == "ollama":
        from ...data_structures.ollama_objects import OllamaJSONResponseSpec

        return OllamaJSONResponseSpec(model=model, instruction=instruction)
    else:
        raise NotImplementedError(
            f"Structured output not implemented for provider '{provider}'"
        )


async def generate_structured_json(
    client: Any,
    messages: list[dict] | str,
    response_model: type[BaseModel],
    model: str = "default",
    **kwargs: Any,
) -> LLMResponseProtocol:
    """
    Generic non-streaming structured output.
    Most users should just call: await client.create_chat(..., response_model=MyModel)
    """
    return await client.create_chat(
        messages=messages,
        model=model,
        response_model=response_model,
        **kwargs,
    )


async def generate_structured_json_stream(
    client: Any,
    messages: list[dict] | str,
    response_model: type[BaseModel],
    model: str = "default",
    **kwargs: Any,
) -> AsyncIterator[tuple[LLMResponseProtocol, LLMStreamingChunkProtocol]]:
    """
    Generic streaming structured output.
    """
    async for item in client.create_chat(
        messages=messages,
        model=model,
        response_model=response_model,
        stream=True,
        **kwargs,
    ):
        yield item
