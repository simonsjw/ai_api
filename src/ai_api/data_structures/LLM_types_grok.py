#! /home/simon/anaconda3/envs/grok/bin/python
# -*- coding: utf-8 -*-

"""
Module for Grok API request structures.

This module provides immutable dataclasses for structuring inputs and requests
to the Grok API or for managing responses. It includes validation for safety and
correctness. The classes for the request are layered as follows:
- `GrokMessage`: Represents a single message with role and content (supports
  multimodal content like text or image lists).
- `GrokInput`: Wraps a sequence of `GrokMessage` instances, representing the
  full conversation input for an API request.
- `GrokRequest`: Defines the full API request, including the `GrokInput`,
  model parameters, tools, and optional structured output schemas.

Flow of `request' use:
1. Create individual `GrokMessage` instances (or use `GrokMessage.from_dict`
   for deserialisation).
2. Combine them into a `GrokInput` (or use `GrokInput.from_list` for
   deserialisation from a list of dicts).
3. Pass the `GrokInput` to `GrokRequest` along with other parameters.
4. Serialise the `GrokRequest` via `to_dict()` for API submission.

The GrokResponse object is used for interpreting a return from the Grok API.

Examples
--------
Basic text request:
    >>> messages = [{"role": "user", "content": "What is 101*3?"}]
    >>> grok_input = GrokInput.from_list(messages)
    >>> request = GrokRequest(input=grok_input)
    >>> api_payload = request.to_dict()

Multimodal with image:
    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": [
    ...             {"type": "input_text", "text": "Describe this image."},
    ...             {
    ...                 "type": "input_image",
    ...                 "image_url": "https://example.com/image.jpg",
    ...                 "detail": "high",
    ...             },
    ...         ],
    ...     }
    ... ]
    >>> grok_input = GrokInput.from_list(messages)
    >>> request = GrokRequest(input=grok_input, temperature=0.5)
    >>> api_payload = request.to_dict()

With tools:
    >>> tools = [
    ...     {
    ...         "type": "function",
    ...         "function": {
    ...             "name": "get_weather",
    ...             "description": "Get current weather",
    ...             "parameters": {
    ...                 "type": "object",
    ...                 "properties": {"city": {"type": "string"}},
    ...             },
    ...         },
    ...     }
    ... ]
    >>> messages = [{"role": "user", "content": "Weather in Sydney?"}]
    >>> grok_input = GrokInput.from_list(messages)
    >>> request = GrokRequest(input=grok_input, tools=tools, tool_choice="auto")
    >>> api_payload = request.to_dict()

With structured schema (using Pydantic):
    >>> from pydantic import BaseModel, Field
    >>> class MathResult(BaseModel):
    ...     value: int = Field(description="The computed result")
    >>> messages = [{"role": "user", "content": "What is 101*3? Use JSON output."}]
    >>> grok_input = GrokInput.from_list(messages)
    >>> request = GrokRequest(input=grok_input, structured_schema=MathResult)
    >>> api_payload = request.to_dict()  # Includes "response_format"

Custom sampling:
    >>> messages = [
    ...     {"role": "system", "content": "Be concise."},
    ...     {"role": "user", "content": "Hello"},
    ... ]
    >>> grok_input = GrokInput.from_list(messages)
    >>> request = GrokRequest(
    ...     input=grok_input,
    ...     model="grok-4-0709",
    ...     temperature=0.2,
    ...     top_p=0.9,
    ...     max_output_tokens=100,
    ...     store=False,
    ... )
    >>> api_payload = request.to_dict()

GrokResponse:
    >>> raw_json = '''
    ... {
    ...   "id": "resp_01J...",
    ...   "created_at": 1736759400,
    ...   "model": "grok-3",
    ...   "output": [
    ...     {
    ...       "type": "message",
    ...       "role": "assistant",
    ...       "content": [{"type": "output_text", "text": "Hello Simon!"}],
    ...       "status": "completed"
    ...     }
    ...   ],
    ...   "usage": {"prompt_tokens": 28, "completion_tokens": 9, "total_tokens": 37}
    ... }
    ... '''
    >>> response = GrokResponse.from_dict(json.loads(raw_json))
    >>> print(response.text)  # "Hello Simon!"
    >>> print(response.tool_calls)  # []
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, Type, cast

from pydantic import BaseModel

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
    immutability for thread-safety. Use to_dict() for serialisation.
    Structured schema maps to 'response_format' for JSON-structured outputs.

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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the request to a dictionary for JSON serialisation.

        Handles schema conversion for structured outputs and omits None values
        for efficiency.

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
