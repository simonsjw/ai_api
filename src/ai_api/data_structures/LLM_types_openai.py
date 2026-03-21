#!/usr/bin/env python3
"""
Custom Typing for OpenAI related Data Structures
---------------------------------------------
This module defines custom types and data models to aid in type checking and debugging
for LLM (Large Language Model) interactions.

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
>>> from LLM_types import OPEN_AI_BATCH_OUTPUT, GrokRequest, GrokInput
>>> # Parse API response into OPEN_AI_BATCH_OUTPUT for type-safe access

"""

from typing import TypedDict


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
    completion_tokens_details: dict[str, int]                                             # Keys: "reasoning_tokens", etc.


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
