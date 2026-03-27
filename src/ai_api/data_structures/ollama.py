#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Ollama API request structures (refined for streaming with common protocol).

This module provides immutable dataclasses for structuring inputs and requests
to the local Ollama API. It mirrors the Grok structure for consistency while
adapting to Ollama's OpenAI-compatible chat format (uses "messages" key, no
"input" wrapper). Validation ensures safety and correctness.

The classes are layered as follows:
- `OllamaMessage`: Represents a single message with role and content (supports
  multimodal content like text or image lists).
- `OllamaInput`: Wraps a sequence of `OllamaMessage` instances, representing
  the full conversation input.
- `OllamaRequest`: Defines the full chat request with model parameters.
- `OllamaResponse`: Represents a response from Ollama.
- `OllamaStreamingChunk`: Refined provider-specific streaming chunk implementing
  LLMStreamingChunkProtocol (richer metadata: finish_reason, partial tool calls,
  is_final flag, raw delta).

No batch data model is included because Ollama does not provide a native batch
API (local inference is handled per-request; parallelism is managed at the
client level via async gather).

Flow of request use:
1. Create individual `OllamaMessage` instances (or use `OllamaMessage.from_dict`
   for deserialisation).
2. Combine them into an `OllamaInput` (or use `OllamaInput.from_list` for
   deserialisation from a list of dicts).
3. Pass the `OllamaInput` to `OllamaRequest` along with other parameters.
4. Serialise the `OllamaRequest` via `to_payload()` (automatically uses
   "messages" key) for API submission or streaming.

Flow of streaming (refined):
1. Pass stream=True to the client.
2. The client yields `OllamaStreamingChunk` objects that implement the common
   LLMStreamingChunkProtocol for uniform consumption across providers.

Examples
--------
Basic text request:
    >>> messages = [{"role": "user", "content": "What is 101*3?"}]
    >>> ollama_input = OllamaInput.from_list(messages)
    >>> request = OllamaRequest(input=ollama_input)
    >>> api_payload = request.to_payload()

Streaming chunk example (refined usage):
    >>> async for chunk in await client.generate(req, stream=True):
    ...     print(chunk.text, end="", flush=True)
    ...     if chunk.is_final:
    ...         print(f"\\nFinished: {chunk.finish_reason}")
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, cast, runtime_checkable


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


type Role = Literal[
    "system", "user", "assistant"
]                                                                                         # Ollama uses a subset of Grok roles.


@dataclass(frozen=True)
class OllamaMessage:
    """
    Immutable structure for individual messages in an Ollama conversation.

    Supports multimodal content (text or mixed text/image lists) to stay
    consistent with GrokMessage.

    Parameters
    ----------
    role : Role
        The role of the message sender (e.g., "user", "assistant").
    content : str | list[dict[str, Any]]
        The content: str for text, or list for multimodal.
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
    def from_dict(cls, msg_dict: dict[str, Any]) -> "OllamaMessage":
        """
        Create an OllamaMessage instance from a dict with validation.

        Validates keys, role, and content (str or multimodal list).

        Parameters
        ----------
        msg_dict : dict[str, Any]
            Input dict with 'role' and 'content' keys.

        Returns
        -------
        OllamaMessage
            Validated immutable instance.

        Raises
        ------
        ValueError
            If keys missing, role invalid, or content malformed.
        """
        required_keys = {"role", "content"}                                               # Set for O(1) checks.
        if not required_keys.issubset(msg_dict.keys()):
            missing = required_keys - set(msg_dict.keys())
            message_string: str = f"Missing keys: {missing}"
            raise ValueError(message_string)

        role_str = msg_dict["role"]
        allowed_roles = ("system", "user", "assistant")                                   # Tuple for fast lookup.
        if role_str not in allowed_roles:
            raise ValueError(
                f"Invalid role '{role_str}'. Must be one of: {', '.join(allowed_roles)}."
            )
        role: Role = cast(Role, role_str)                                                 # Narrow type post-validation.

        content = msg_dict["content"]
        if isinstance(content, str):                                                      # Text case.
            pass
        elif isinstance(content, list):                                                   # Multimodal validation.
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError("List items in content must be dicts.")
                typ = item.get("type")
                if typ == "text":
                    if "text" not in item or not isinstance(item["text"], str):
                        raise ValueError("text requires 'text': str.")
                elif typ == "image_url":
                    if "image_url" not in item or not isinstance(
                        item["image_url"], str
                    ):
                        raise ValueError("image_url requires 'image_url': str.")
                elif typ is None:
                    raise ValueError("Each item must have 'type' key.")
                else:
                    raise ValueError(f"Invalid type '{typ}' in content item.")
        else:
            raise ValueError("Content must be str or list[dict].")

        return cls(role=role, content=content)                                            # Type-safe creation.


@dataclass(frozen=True)
class OllamaInput:
    """
    Immutable structure for a sequence of messages in an Ollama conversation.

    Represents the 'messages' field used by Ollama (and OpenAI-compatible APIs).

    Parameters
    ----------
    messages : Tuple[OllamaMessage, ...]
        The sequence of messages in the conversation.
    """

    messages: tuple[OllamaMessage, ...]

    def to_list(self) -> list[dict[str, Any]]:
        """
        Convert the conversation to a list of dictionaries for JSON
        serialisation (e.g., for the 'messages' field).

        Returns
        -------
        list[dict[str, Any]]
            List of dictionaries, each with 'role' and 'content'.
        """
        return [msg.to_dict() for msg in self.messages]                                   # Efficient list comprehension.

    @classmethod
    def from_list(cls, msg_list: list[dict[str, Any]]) -> "OllamaInput":
        """
        Create an OllamaInput instance from a list of dicts with validation.

        Validates each message using OllamaMessage.from_dict.

        Parameters
        ----------
        msg_list : list[dict[str, Any]]
            List of input dicts, each with 'role' and 'content'.

        Returns
        -------
        OllamaInput
            Validated immutable instance.

        Raises
        ------
        ValueError
            If any message is invalid.
        """
        ollama_messages = [
            OllamaMessage.from_dict(msg_dict) for msg_dict in msg_list
        ]                                                                                 # Efficient validation loop.
        return cls(tuple(ollama_messages))                                                # Tuple for immutability.


@dataclass(frozen=True)
class OllamaRequest:
    """
    Immutable request structure for Ollama chat completions.

    Uses "messages" key (Ollama/OpenAI style) for full consistency with
    OllamaInput. No batch support (Ollama has none).

    Parameters
    ----------
    input : OllamaInput
        The input messages (required).
    model : str, optional
        Model name (e.g., "qwen3-coder-next:latest").
    temperature : float | None, optional
        Sampling temperature (default None lets Ollama use its own default).
    top_p : float | None, optional
        Nucleus sampling parameter.
    max_tokens : int | None, optional
        Maximum tokens in output (Ollama uses 'num_predict' internally).
    """

    input: OllamaInput
    model: str = "llama3.2"
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None

    def to_payload(self) -> dict[str, Any]:
        """
        Convert the request to the exact dictionary required by Ollama
        (uses "messages" key).

        This method is used by the LLMClient for automatic payload inference.

        Returns
        -------
        dict[str, Any]
            Dictionary ready for Ollama /api/chat or /api/generate.
        """
        result: dict[str, Any] = {
            "model": self.model,
            "messages": self.input.to_list(),
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens is not None:
            result["options"] = {"num_predict": self.max_tokens}                          # Ollama option.

        return {k: v for k, v in result.items() if v is not None}                         # Omit None values.

    def get_endpoint(self) -> str:
        """
        Return the endpoint path used by this request (for Ollama chat).

        Returns
        -------
        str
            "/api/chat"
        """
        return "/api/chat"                                                                # Fixed for Ollama chat API.


@dataclass(frozen=True)
class OllamaResponse:
    """
    Representation of a response from the Ollama chat API.

    Mirrors GrokResponse for consistency across the library.

    Parameters
    ----------
    model : str
        Model that produced the response.
    message : dict[str, Any]
        The assistant message (contains role and content).
    done : bool
        Whether the generation is complete.
    total_duration : int | None, optional
        Total time taken (nanoseconds).
    """

    model: str
    message: dict[str, Any]
    done: bool
    total_duration: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """
        Extract the plain text content from the assistant message.

        Returns
        -------
        str
            Stripped assistant text.
        """
        return self.message.get("content", "").strip()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OllamaResponse":
        """
        Create an OllamaResponse instance from raw Ollama JSON.

        Parameters
        ----------
        data : dict[str, Any]
            Parsed JSON from Ollama /api/chat response.

        Returns
        -------
        OllamaResponse
            Validated immutable response.
        """
        return cls(
            model=str(data["model"]),
            message=data.get("message", {}),
            done=bool(data.get("done", False)),
            total_duration=data.get("total_duration"),
            raw=data.copy(),
        )                                                                                 # Copy raw for debugging; direct field mapping.


@dataclass(frozen=True)
class OllamaStreamingChunk:
    """
    Refined immutable streaming chunk for Ollama chat API.

    Implements LLMStreamingChunkProtocol for uniform handling across providers.
    Supports partial text, tool calls, finish_reason, and final-chunk detection.

    Parameters
    ----------
    model : str
        Model that produced the chunk.
    message : dict[str, Any]
        The partial assistant message (contains role and content delta).
    done : bool
        Whether this chunk indicates completion.
    raw : dict[str, Any]
        Original chunk for debugging / persistence.
    """

    model: str
    message: dict[str, Any]
    done: bool
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """
        Extract plain text from the partial message.

        Returns
        -------
        str
            Stripped text content or empty string.
        """
        return self.message.get("content", "").strip()

    @property
    def finish_reason(self) -> str | None:
        """
        Return finish_reason if present (Ollama sets this on final chunk).

        Returns
        -------
        str | None
            e.g. "stop", None if still streaming.
        """
        return self.message.get("finish_reason") if self.done else None                   # type: ignore[return-value]

    @property
    def tool_calls_delta(self) -> list[dict[str, Any]] | None:
        """
        Return partial tool calls if present in this chunk (Ollama tool support).

        Returns
        -------
        list[dict[str, Any]] | None
            List of tool call deltas or None.
        """
        return self.message.get("tool_calls")                                             # type: ignore[return-value]

    @property
    def is_final(self) -> bool:
        """
        True if this is the final chunk (done flag set by Ollama).

        Returns
        -------
        bool
            Final-chunk flag for easy loop termination.
        """
        return self.done

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OllamaStreamingChunk":
        """
        Create OllamaStreamingChunk from raw Ollama streaming JSON.

        Parameters
        ----------
        data : dict[str, Any]
            Raw chunk from Ollama streaming response.

        Returns
        -------
        OllamaStreamingChunk
            Validated immutable chunk implementing LLMStreamingChunkProtocol.
        """
        return cls(
            model=str(data.get("model", "")),
            message=data.get("message", {}),
            done=bool(data.get("done", False)),
            raw=data.copy(),
        )                                                                                 # Direct field mapping; handles missing keys gracefully.
