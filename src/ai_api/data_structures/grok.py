#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Grok API request structures (refined for streaming with common protocol).

This module provides immutable dataclasses for structuring inputs and requests
to the Grok API or for managing responses. It includes validation for safety and
correctness. The classes for the request are layered as follows:

- `GrokMessage`: Represents a single message with role and content (supports
  multimodal content like text or image lists).
- `GrokInput`: Wraps a sequence of `GrokMessage` instances, representing the
  full conversation input for an API request.
- `GrokRequest`: Defines the full API request, including the `GrokInput`,
  model parameters, tools, and optional structured output schemas.
- `GrokBatchRequest`: High-level batch container for submitting multiple
  `GrokRequest` objects to Grok's /v1/batches endpoint.
- `GrokBatchResponse`: Represents the asynchronous batch result returned by
  Grok.
- `GrokStreamingChunk`: Refined provider-specific streaming chunk implementing
  LLMStreamingChunkProtocol (richer metadata: finish_reason, partial tool calls,
  is_final flag, raw delta).

Flow of request use (single or streaming):
1. Create individual `GrokMessage` instances (or use `GrokMessage.from_dict`
   for deserialisation).
2. Combine them into a `GrokInput` (or use `GrokInput.from_list` for
   deserialisation from a list of dicts).
3. Pass the `GrokInput` to `GrokRequest` along with other parameters.
4. Serialise the `GrokRequest` via `to_payload()` (automatically uses "input"
   for the Responses API) for API submission or streaming.

Flow of batch use:
1. Build one or more `GrokRequest` objects.
2. Pass the tuple of requests to `GrokBatchRequest`.
3. Call `to_payload()` to obtain the exact JSON body for /v1/batches.

Flow of streaming (refined):
1. Pass stream=True to the client.
2. The client yields `GrokStreamingChunk` objects that implement the common
   LLMStreamingChunkProtocol for uniform consumption across providers.

The GrokResponse object is used for interpreting a return from the Grok API
(single or within a batch).

Examples
--------
Basic text request:
    >>> messages = [{"role": "user", "content": "What is 101*3?"}]
    >>> grok_input = GrokInput.from_list(messages)
    >>> request = GrokRequest(input=grok_input)
    >>> api_payload = request.to_payload()

Batch of three requests:
    >>> req1 = GrokRequest(input=GrokInput.from_list([{"role": "user", "content": "Hello"}]))
    >>> req2 = GrokRequest(
    ...     input=GrokInput.from_list([{"role": "user", "content": "Weather in Sydney?"}])
    ... )
    >>> req3 = GrokRequest(
    ...     input=GrokInput.from_list([{"role": "user", "content": "Explain Rust lifetimes"}])
    ... )
    >>> batch = GrokBatchRequest(requests=(req1, req2, req3), completion_window="24h")
    >>> batch_payload = batch.to_payload()

Streaming chunk example (refined usage):
    >>> async for chunk in await client.generate(req, stream=True):
    ...     print(chunk.text, end="", flush=True)
    ...     if chunk.is_final:
    ...         print(f"\\nFinished: {chunk.finish_reason}")
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, Type, cast, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class LLMStreamingChunkProtocol(Protocol):
    """Common contract for all streaming chunks (Grok, Ollama, future providers).

    Ensures uniform handling in LLMClient and higher-level code while preserving
    provider-specific details. All properties are efficient O(1) getters.
    """

    @property
    def text(self) -> str: ...
    @property
    def finish_reason(self) -> str | None: ...
    @property
    def tool_calls_delta(self) -> list[dict[str, Any]] | None: ...
    @property
    def is_final(self) -> bool: ...
    @property
    def raw(self) -> dict[str, Any]: ...


type SaveMode = Literal["none", "json_files", "postgres"]

type Role = Literal[
    "system", "user", "assistant", "developer"
]                                                                                         # Type alias for roles; "developer" aliases "system".


@dataclass(frozen=True)
class GrokMessage:
    """
    Immutable structure for individual messages in a Grok conversation.

    Supports multimodal content (text or mixed text/image lists).

    Parameters
    ----------
    role : Role
        The role of the message sender (e.g., "user", "assistant").
    content : str | list[dict[str, Any]]
        The content: str for text, or list for multimodal (e.g., input_text
        and input_image items).
    """

    role: Role
    content: str | list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the message to a dictionary for JSON serialisation.

        Returns
        -------
        dict[str, Any]
            Dictionary with 'role' and 'content'.
        """
        return {
            "role": self.role,
            "content": self.content,
        }                                                                                 # Direct return; handles str or list.

    @classmethod
    def from_dict(cls, msg_dict: dict[str, Any]) -> "GrokMessage":
        """
        Create a GrokMessage instance from a dict with validation.

        Validates keys, role, and content (str or multimodal list). For lists,
        checks each item's type, required fields, and constraints.

        Parameters
        ----------
        msg_dict : dict[str, Any]
            Input dict with 'role' and 'content' keys.

        Returns
        -------
        GrokMessage
            Validated immutable instance.

        Raises
        ------
        ValueError
            If keys missing, role invalid, content malformed, or types mismatch.
        """
        required_keys = {"role", "content"}                                               # Set for O(1) checks.
        if not required_keys.issubset(msg_dict.keys()):
            missing = required_keys - set(msg_dict.keys())
            message_string: str = f"Missing keys: {missing}"
            raise ValueError(message_string)

        role_str = msg_dict["role"]
        allowed_roles = (
            "system",
            "user",
            "assistant",
            "developer",
        )                                                                                 # Tuple for fast 'in' check.
        if role_str not in allowed_roles:
            raise ValueError(
                f"Invalid role '{role_str}'. Must be one of: {', '.join(allowed_roles)}."
            )
        role: Role = cast(Role, role_str)                                                 # Narrow type post-validation.

        content = msg_dict["content"]
        if isinstance(content, str):                                                      # Text case: simple check.
            pass
        elif isinstance(content, list):                                                   # Multimodal: validate each item.
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError("List items in content must be dicts.")
                typ = item.get("type")
                if typ == "input_text":
                    if "text" not in item or not isinstance(item["text"], str):
                        raise ValueError("input_text requires 'text': str.")
                elif typ == "input_image":
                    if "image_url" not in item or not isinstance(
                        item["image_url"], str
                    ):
                        raise ValueError("input_image requires 'image_url': str.")
                    detail = item.get("detail", "auto")
                    if detail not in ("auto", "low", "high"):
                        raise ValueError("detail must be 'auto', 'low', or 'high'.")
                elif typ is None:
                    raise ValueError("Each item must have 'type' key.")
                else:
                    raise ValueError(f"Invalid type '{typ}' in content item.")
        else:
            raise ValueError("Content must be str or list[dict].")

        return cls(role=role, content=content)                                            # Type-safe creation.


@dataclass(frozen=True)
class GrokInput:
    """
    Immutable structure for a sequence of messages in a Grok conversation.

    This represents the 'input' field for API requests as an array of messages.

    Parameters
    ----------
    messages : Tuple[GrokMessage, ...]
        The sequence of messages in the conversation.
    """

    messages: tuple[GrokMessage, ...]

    def to_list(self) -> list[dict[str, Any]]:
        """
        Convert the conversation to a list of dictionaries for JSON
        serialisation (e.g., for the 'input' field in API requests).

        Returns
        -------
        list[dict[str, Any]]
            List of dictionaries, each with 'role' and 'content'.
        """
        return [msg.to_dict() for msg in self.messages]                                   # Efficient list comprehension.

    @classmethod
    def from_list(cls, msg_list: list[dict[str, Any]]) -> "GrokInput":
        """
        Create a GrokInput instance from a list of dicts with validation.

        Validates each message using GrokMessage.from_dict.

        Parameters
        ----------
        msg_list : list[dict[str, Any]]
            List of input dicts, each with 'role' and 'content'.

        Returns
        -------
        GrokInput
            Validated immutable instance.

        Raises
        ------
        ValueError
            If any message is invalid.
        """
        grok_messages = [
            GrokMessage.from_dict(msg_dict) for msg_dict in msg_list
        ]                                                                                 # Efficient validation loop.
        return cls(tuple(grok_messages))                                                  # Tuple for immutability.


@dataclass(frozen=True)
class GrokRequest:
    """
    Immutable request structure for Grok Responses API (POST /v1/responses).

    This class defines parameters for creating a new response, ensuring
    immutability for thread-safety. Use to_payload() for serialisation
    (automatically uses "input" key for the Responses API).

    Parameters
    ----------
    input : GrokInput
        The input messages (required; array of role/content dicts, supports
        multimodal content like images).
    model : str, optional
        Model name (e.g., "grok-beta", "grok-4-0709"; no default in API, but
        "grok-beta" used here for common cases).
    temperature : float | None, optional
        Sampling temperature (controls randomness; API default unspecified,
        typical range 0.0-2.0).
    top_p : float | None, optional
        Nucleus sampling parameter (alternative to temperature; range 0.0-1.0).
    max_output_tokens : int | None, optional
        Maximum tokens in output (limits length; nullable).
    store : bool, optional
        Whether to store input/response for retrieval (default: True; stored
        for 30 days).
    tools : list[dict[str, Any]] | None, optional
        List of tools (JSON schema; max 128; supports functions, web search).
    tool_choice : str | dict[str, Any] | None, optional
        Controls tool calling ("auto", "none", "required", or specific tool
        dict).
    parallel_tool_calls : bool | None, optional
        Enables parallel tool calls (default: True in API responses).
    structured_schema : dict[str, Any] | Type[BaseModel] | None, optional
        JSON schema dict or Pydantic model for structured outputs. Serialised
        to 'response_format' as {"type": "json_object", "schema": ...}.
    """

    input: GrokInput
    model: str = "grok-3-mini"
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    store: bool = True
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    structured_schema: dict[str, Any] | Type[BaseModel] | None = None

    def to_payload(self) -> dict[str, Any]:
        """
        Convert the request to the exact dictionary required by the Grok
        Responses API (uses "input" key).

        Handles schema conversion for structured outputs and omits None values
        for efficiency. This method is used by the LLMClient for automatic
        payload inference.

        Returns
        -------
        dict[str, Any]
            Dictionary for API submission.

        Raises
        ------
        ValueError
            If structured_schema invalid.
        """
        result: dict[str, Any] = {                                                        # Base dict; efficient init.
            "input": self.input.to_list(),
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "store": self.store,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
        }

        if self.structured_schema is not None:                                            # Add response_format if present.
            result["response_format"] = {
                "type": "json_object",
                "schema": self._serialize_schema(),
            }

        return {
            k: v for k, v in result.items() if v is not None
        }                                                                                 # Omit None for clean payload.

    def _serialize_schema(self) -> dict[str, Any]:
        """
        Helper to serialise structured_schema.

        Converts Pydantic class to JSON schema if applicable.

        Returns
        -------
        dict[str, Any]
            Schema dict.

        Raises
        ------
        ValueError
            If invalid type.
        """
        if isinstance(self.structured_schema, dict):
            return self.structured_schema                                                 # Direct return.
        if (
            BaseModel is not None
            and isinstance(self.structured_schema, type)
            and issubclass(self.structured_schema, BaseModel)
        ):
            schema_class = cast(Type[BaseModel], self.structured_schema)
            return schema_class.model_json_schema()                                       # Efficient classmethod.
        raise ValueError("structured_schema must be dict or BaseModel subclass.")

    def get_endpoint(self) -> str:
        """
        Return the endpoint path used by this request (for Responses API).

        Returns
        -------
        str
            "/v1/responses"
        """
        return "/v1/responses"                                                            # Fixed for Grok Responses API.


@dataclass(frozen=True)
class GrokBatchRequest:
    """
    Immutable batch request for Grok's /v1/batches endpoint.

    Holds multiple GrokRequest objects (each becomes one line in the batch).
    Efficient tuple storage; generates exact payload for the batch API.

    Parameters
    ----------
    requests : Tuple[GrokRequest, ...]
        The individual requests to batch (one or more).
    completion_window : str, optional
        When the batch should complete ("24h" default, Grok supported value).
    """

    requests: tuple[GrokRequest, ...]
    completion_window: str = "24h"

    def to_payload(self) -> dict[str, Any]:
        """
        Convert the batch to the exact dictionary required by Grok /v1/batches.

        Each request becomes a batch line with custom_id and body.

        Returns
        -------
        dict[str, Any]
            Ready-to-POST payload (endpoint, requests list, completion_window).
        """
        batch_lines = []
        for i, req in enumerate(self.requests):
            batch_lines.append(
                {
                    "custom_id": f"req-{i}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": req.to_payload(),                                             # Use provider-specific payload.
                }
            )
        return {
            "endpoint": "/v1/responses",
            "requests": batch_lines,
            "completion_window": self.completion_window,
        }                                                                                 # Direct dict construction; O(n) but n is small for batches.


@dataclass(frozen=True)
class GrokResponse:
    """
    Representation of a response from the xAI Grok Chat Completions API.
    """

    id: str
    created_at: int
    model: str
    output: Sequence[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    status: Literal["completed", "in_progress", "incomplete"] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        parts: list[str] = []
        for item in self.output:
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    parts.append(part.get("text", ""))
        return "".join(parts).strip()

    @property
    def first_message(self) -> dict[str, Any] | None:
        for item in self.output:
            if item.get("type") == "message":
                return item
        return None

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for item in self.output:
            t = item.get("type")
            if t in (
                "function_call",
                "web_search_call",
                "x_search_call",
                "code_interpreter_call",
                "file_search_call",
                "mcp_call",
            ):
                calls.append(item)
        return calls

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GrokResponse":
        """
        Create a GrokResponse instance from a raw API response dictionary.

        This is the recommended way to instantiate GrokResponse from real API data.

        Parameters
        ----------
        data : dict[str, Any]
            The parsed JSON response body from the chat completions endpoint.

        Returns
        -------
        GrokResponse
            A frozen dataclass instance populated from the API data.

        Raises
        ------
        KeyError
            If required top-level fields ('id', 'created_at', 'model') are missing.
        TypeError
            If critical fields have the wrong type (very basic coercion only).
        """
        # Keep the original data for debugging / forward compatibility
        raw = data.copy()

        # Required fields – raise early if missing
        try:
            response_id = str(data["id"])
            created_at = int(data["created_at"])
            model = str(data["model"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Missing or invalid required field(s) in API response: "
                "'id', 'created_at', 'model'"
            ) from exc

        # Optional / nullable fields
        output = data.get("output", [])
        if not isinstance(output, list):
            output = []

        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            usage = None

        status = data.get("status")
        if status not in ("completed", "in_progress", "incomplete", None):
            status = None

        return cls(
            id=response_id,
            created_at=created_at,
            model=model,
            output=output,
            usage=usage,
            status=status,
            raw=raw,
        )

    def __str__(self) -> str:
        txt = self.text[:80] + "…" if len(self.text) > 80 else self.text
        tc = len(self.tool_calls)
        return f"GrokResponse(id={self.id}, model={self.model}, text={txt!r}, tool_calls={tc})"


@dataclass(frozen=True)
class GrokBatchResponse:
    """
    Immutable representation of a completed Grok batch result.

    Contains the original batch_id and a list of per-request results
    (each a GrokResponse or error dict).

    Parameters
    ----------
    batch_id : str
        Unique Grok batch identifier.
    results : Tuple[dict[str, Any], ...]
        List of outcome dicts (one per request; contains GrokResponse data or error).
    """

    batch_id: str
    results: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GrokBatchResponse":
        """
        Create GrokBatchResponse from the raw /v1/batches/{id} response.

        Parameters
        ----------
        data : dict[str, Any]
            Parsed JSON from Grok batch status endpoint.

        Returns
        -------
        GrokBatchResponse
            Validated immutable batch result.
        """
        return cls(
            batch_id=str(data["id"]),
            results=tuple(data.get("results", [])),
        )                                                                                 # Tuple for immutability; direct conversion.


@dataclass(frozen=True)
class GrokStreamingChunk:
    """
    Refined immutable streaming chunk for Grok Responses API (delta output).

    Implements LLMStreamingChunkProtocol for uniform handling across providers.
    Supports partial text, tool calls, finish_reason, and final-chunk detection.

    Parameters
    ----------
    id : str
        Chunk identifier.
    delta : dict[str, Any] | None
        Delta payload (contains content blocks, tool_calls, etc.).
    raw : dict[str, Any]
        Original chunk for debugging / persistence.
    """

    id: str
    delta: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """
        Extract plain text from the delta (if present).

        Returns
        -------
        str
            Stripped text content or empty string.
        """
        if not self.delta:
            return ""
        content = self.delta.get("content", [])
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if part.get("type") == "output_text":
                    parts.append(part.get("text", ""))
            return "".join(parts).strip()
        return str(content).strip()

    @property
    def finish_reason(self) -> str | None:
        """
        Return the finish_reason if present in this chunk (only on final chunk).

        Returns
        -------
        str | None
            e.g. "stop", "length", None if still streaming.
        """
        return self.delta.get("finish_reason") if self.delta else None                    # type: ignore[union-attr]

    @property
    def tool_calls_delta(self) -> list[dict[str, Any]] | None:
        """
        Return partial tool calls if present in this chunk.

        Returns
        -------
        list[dict[str, Any]] | None
            List of tool call deltas or None.
        """
        if not self.delta:
            return None
        return self.delta.get("tool_calls")                                               # type: ignore[return-value]

    @property
    def is_final(self) -> bool:
        """
        True if this is the final chunk (finish_reason present or usage emitted).

        Returns
        -------
        bool
            Final-chunk flag for easy loop termination.
        """
        return bool(self.finish_reason) or bool(self.delta and self.delta.get("usage"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GrokStreamingChunk":
        """
        Create GrokStreamingChunk from raw streaming delta (Grok Responses API).

        Parameters
        ----------
        data : dict[str, Any]
            Raw chunk from Grok streaming response.

        Returns
        -------
        GrokStreamingChunk
            Validated immutable chunk implementing LLMStreamingChunkProtocol.
        """
        return cls(
            id=str(data.get("id", "")),
            delta=data.get("delta"),
            raw=data.copy(),
        )                                                                                 # Direct field mapping; handles missing keys gracefully.
