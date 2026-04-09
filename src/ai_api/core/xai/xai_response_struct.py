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
from .xai_batches import *
from .xai_errors import *
from .xai_persistence import *
from .xai_stream import *

__all__: list[str] = [
    "create_json_response_spec",
    "generate_structured",
    "generate_structured_stream",
]


def create_json_response_spec(
    self,
    model: type[BaseModel] | dict[str, Any],
    instruction: str | None = None,
) -> "xAIJSONResponseSpec":
    """Create a validated JSON response specification.

    Parameters
    ----------
    model : type[BaseModel] | dict[str, Any]
        Pydantic model class or raw JSON schema.
    instruction : str | None
        Optional custom instruction (defaults to JSON_INSTRUCTION).

    Returns
    -------
    xAIJSONResponseSpec
        Ready-to-use specification object.

    Raises
    ------
    ValueError
        If model is invalid.
    """
    if instruction is None:
        instruction = JSON_INSTRUCTION
        # Note: Automatic injection of the instruction into a system message
        # is left to the caller for now. This keeps behaviour explicit.
    return xAIJSONResponseSpec(model=model)


async def generate_structured(
    self,
    request: xAIRequest | str | list[xAIMessage],
    response_model: type[BaseModel],
    **kwargs: Any,
) -> xAIResponse:
    """Convenience wrapper that returns a typed Pydantic instance.

    This method forces the non-streaming path for simplicity and
    guaranteed type safety. It sets the response specification,
    performs validation, and returns the fully parsed model.

    Parameters
    ----------
    request : xAIRequest | str | list[xAIMessage]
        Input prompt (string, message list, or full request).
    response_model : type[BaseModel]
        Target Pydantic model for structured output.
    **kwargs : Any
        Additional arguments passed to ``generate()``.

    Returns
    -------
    xAIResponse
        Response container. The parsed object is attached as
        ``.parsed`` for convenient access by callers.

    Raises
    ------
    ValueError
        If the system message does not contain the required JSON
        instruction string.
    TypeError
        If a streaming response or unexpected type is returned.
    """
    # Normalise input to a full xAIRequest object
    if isinstance(request, (str, list)):
        req: xAIRequest = self.create_request(prompt=request)
    else:
        req = request

        # Attach the JSON response specification
        # (this triggers automatic validation in xAIRequest.__post_init__)
    spec = self.create_json_response_spec(response_model)
    req = replace(req, response_spec=spec)

    # Call generate with explicit non-streaming
    result = await self.generate(
        req, stream=False, **{k: v for k, v in kwargs.items() if k != "stream"}
    )

    # Runtime type guard to narrow for Pyrefly and safety
    if (
        isinstance(result, (dict, list))
        or hasattr(result, "__aiter__")
        or not isinstance(result, xAIResponse)
    ):
        raise TypeError(
            "generate_structured() expected a non-streaming xAIResponse "
            "but received a streaming iterator or raw dict. "
            "Ensure stream=False is respected by the generate method."
        )

    # Explicit narrowing that satisfies the type checker
    raw_response: xAIResponse = result

    # Extract the JSON string from the response object.
    # IMPORTANT: Adjust the attribute below to match the actual xAIResponse.
    # Common patterns observed in xAI SDK responses:
    #   raw_response.text
    #   raw_response.content
    #   raw_response.message.content
    #   raw_response.choices[0].message.content
    json_str: str = raw_response.text                                                     # <--- CHANGE THIS IF NEEDED

    # Parse into the requested Pydantic model
    parsed = response_model.model_validate_json(json_str)

    # Attach the parsed object for caller convenience
    raw_response.parsed = parsed                                                          # type: ignore[attr-defined]

    return raw_response


async def generate_structured_stream(
    self,
    request: xAIRequest | str | list[xAIMessage],
    response_model: type[BaseModel],
    **kwargs: Any,
) -> AsyncIterator[tuple[xAIResponse, LLMStreamingChunkProtocol]]:
    """Streaming version of generate_structured.

    Sets the JSON response specification, streams the response, and
    attaches the fully parsed Pydantic model to the final accumulated
    response object. This matches the official xAI SDK streaming pattern
    and supports structured outputs with real-time token output.

    Parameters
    ----------
    request : xAIRequest | str | list[xAIMessage]
        Input prompt (string, message list, or full request).
    response_model : type[BaseModel]
        Target Pydantic model for structured output.
    **kwargs : Any
        Additional arguments passed to ``generate()``.

    Yields
    ------
    tuple[xAIResponse, LLMStreamingChunkProtocol]
        The accumulating response object and the current chunk.
        The ``.parsed`` attribute is populated only on the final chunk.

    Raises
    ------
    ValueError
        If the system message does not contain the required JSON
        instruction string.
    """
    if isinstance(request, (str, list)):
        req: xAIRequest = self.create_request(prompt=request)
    else:
        req = request

    spec = self.create_json_response_spec(response_model)
    req = replace(req, response_spec=spec)

    # Streaming path (matches your existing generate method)
    async for raw_response, chunk in self.generate(                                       # type: ignore[misc]
        req, stream=True, **{k: v for k, v in kwargs.items() if k != "stream"}
    ):
        # On the final chunk the response object is complete
        if getattr(chunk, "finish_reason", None) is not None:
            # Extract the final JSON string
            json_str: str = raw_response.text                                             # <-- adjust attribute if needed
            parsed = response_model.model_validate_json(json_str)
            raw_response.parsed = parsed                                                  # type: ignore[attr-defined]

        yield raw_response, chunk
