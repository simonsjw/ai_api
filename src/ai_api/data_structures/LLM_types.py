#!/usr/bin/env python3
"""
Custom Typing for LLM-Related Data Structures
---------------------------------------------
This module defines custom types and data models to aid in type checking and debugging
for LLM (Large Language Model) interactions. It includes typed dictionaries for OpenAI
and Grok API responses, as well as dataclasses for Grok requests and responses.

These types ensure structural consistency when parsing JSON responses from APIs,
facilitating safer data handling and integration with tools like mypy or pyright.

Notes
-----
- TypedDicts are used for JSON-like structures to enforce key presence and types at
  type-checking time (not runtime).
- Dataclasses provide immutable (frozen) request/response objects with auto-generated
  methods for equality, hashing, and representation.
- TypeAlias is used for complex nested types like batch outputs.
- All types are designed to match API schemas closely for minimal transformation.

Examples
--------
>>> from LLM_types import OPEN_AI_BATCH_OUTPUT, GrokRequest, GrokMessage
>>> req = GrokRequest(messages=[GrokMessage(role="user", content="Hello")])
>>> # Parse API response into OPEN_AI_BATCH_OUTPUT for type-safe access
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

## OpenAI API Types
# -----------------


class OPEN_AI_MESSAGE(TypedDict):
    """
    Typed dictionary for a single message in an OpenAI conversation.
    """

    role: str
    content: str
    refusal: str | None


class OPEN_AI_CHOICE(TypedDict):
    """
    Typed dictionary for a choice in an OpenAI completion response.
    """

    index: int
    message: OPEN_AI_MESSAGE
    logprobs: int | None
    finish_reason: str


class OPEN_AI_USAGE(TypedDict):
    """
    Typed dictionary for token usage statistics in an OpenAI response.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: dict[str, int]  # Keys: "reasoning_tokens", etc.


class OPEN_AI_BODY(TypedDict):
    """
    Typed dictionary for the body of an OpenAI completion response.
    """

    id: str
    object: str
    created: int
    model: str
    choices: list[OPEN_AI_CHOICE]
    usage: OPEN_AI_USAGE
    system_fingerprint: str


class OPEN_AI_RESPONSE(TypedDict):
    """
    Typed dictionary for the full OpenAI API response including status.
    """

    status_code: int
    request_id: str
    body: OPEN_AI_BODY


class OPEN_AI_PROMPT_OUTPUT(TypedDict):
    """
    Typed dictionary for a single prompt output in a batch.
    """

    id: str
    custom_id: str
    response: OPEN_AI_RESPONSE
    error: str | None


# Type alias for a batch of OpenAI prompt outputs
type OPEN_AI_BATCH_OUTPUT = list[OPEN_AI_PROMPT_OUTPUT]

## Grok API Types
# ---------------

type SaveMode = Literal[
    "none", "json_files", "postgres"
]  # Type alias for persistence modes in Grok API client.
type Role = Literal[
    "system", "user", "assistant"
]  # Type alias for the role a prompt is intended to inform.


class GrokMessage(TypedDict):
    """
    Typed dictionary for individual messages in a Grok conversation.
    """

    role: Role
    content: str


@dataclass(frozen=True)
class GrokRequest:
    """
    Dataclass representing a request to the Grok API.

    Parameters
    ----------
    messages : list[GrokMessage]
        List of messages in the conversation.
    model : str, optional
        The model to use (default: "grok-beta").
    temperature : float, optional
        Sampling temperature (default: 1.0).
    max_tokens : Optional[int], optional
        Maximum tokens in response (default: None).
    user_id : Optional[str], optional
        User identifier for tracking (default: None).
    metadata : Optional[dict[str, Any]], optional
        Additional metadata (default: None).
    request_id : Optional[uuid.UUID], optional
        Unique UUID for the request, generated if not provided (default: None).
    """

    messages: list[GrokMessage]
    model: str = "grok-beta"
    temperature: float = 1.0
    max_tokens: int | None = None
    user_id: str | None = None
    metadata: dict[str, Any] | None = None
    request_id: uuid.UUID | None = field(
        default=None
    )  # Optional, set via dataclasses.replace if needed

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the request to a dictionary for JSON serialisation.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the request.
        """
        return {
            "messages": self.messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class GrokResponse:
    """
    Dataclass representing a response from the Grok API.

    Parameters
    ----------
    request : GrokRequest
        The original request.
    content : str
        The generated content from the model.
    response_id : str
        Unique string ID (UUID format) for the response.
    finish_reason : str
        Reason the generation finished (e.g., "stop").
    usage : dict[str, Any]
        Usage statistics (e.g., tokens used).
    raw : dict[str, Any]
        Raw API response dictionary.
    """

    request: GrokRequest
    content: str
    response_id: str
    finish_reason: str
    usage: dict[str, Any]
    raw: dict[str, Any]


#  LocalWords:  settingsDict params ResolvedSettingsDict postgres GrokRequest
#  LocalWords:  GrokMessage
