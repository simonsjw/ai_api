"""
This module provides an interface for structured response functionality
using JSON or Pydantic models with the sync or async client based on the
xAI SDK.

Public methods
--------------
 - create_json_response_spec
   Create a valid JSON Response specification.
 - generate_structured
   Wrapper for a typed Pydantic instance for turn based chats
 - generate_structured_stream
   Wrapper for a typed Pydantic instance for streaming chats
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, AsyncIterator

from pydantic import BaseModel

from ...data_structures.xai_objects import (
    JSON_INSTRUCTION,
    LLMStreamingChunkProtocol,
    xAIInput,
    xAIJSONResponseSpec,
    xAIMessage,
    xAIRequest,
    xAIResponse,
)
from .errors_xai import *
from .persistence_xai import *

__all__: list[str] = [
    "create_json_response_spec",
    "generate_structured_json",
    "generate_structured_json_stream",
]


def create_json_response_spec(
    self,
    model: type[BaseModel] | dict[str, Any],
    instruction: str | None = None,
) -> "xAIJSONResponseSpec":
    if instruction is None:
        instruction = JSON_INSTRUCTION
    return xAIJSONResponseSpec(model=model)


async def generate_structured_json(
    self,
    request: xAIRequest | str | list[xAIMessage],
    response_model: type[BaseModel],
    **kwargs: Any,
) -> xAIResponse:
    """Returns a fully typed Pydantic instance (non-streaming)."""

    if isinstance(request, (str, list)):
        req: xAIRequest = self.create_request(prompt=request)
    else:
        req = request

    spec = xAIJSONResponseSpec(model=response_model)
    req = replace(req, response_spec=spec)

    # Now uses SDK under the hood via to_sdk_chat_kwargs()
    result = await self.generate(req, stream=False, **kwargs)

    # SDK already enforces schema; your parsing step remains for attaching .parsed
    raw_response: xAIResponse = result
    parsed = response_model.model_validate_json(raw_response.text)
    raw_response.set_parsed(parsed)

    return raw_response


async def generate_structured_json_stream(
    self,
    request: xAIRequest | str | list[xAIMessage],
    response_model: type[BaseModel],
    **kwargs: Any,
) -> AsyncIterator[tuple[xAIResponse, LLMStreamingChunkProtocol]]:
    """Streaming structured output (parsed model attached on final chunk)."""

    if isinstance(request, (str, list)):
        req: xAIRequest = self.create_request(prompt=request)
    else:
        req = request

    spec = self.create_json_response_spec(response_model)
    req = replace(req, response_spec=spec)

    async for raw_response, chunk in self.generate(
        req, stream=True, **{k: v for k, v in kwargs.items() if k != "stream"}
    ):
        if getattr(chunk, "finish_reason", None) is not None:
            json_str: str = raw_response.text
            parsed = response_model.model_validate_json(json_str)
            raw_response.parsed = parsed                                                  # type: ignore[attr-defined]

        yield raw_response, chunk
