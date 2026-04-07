#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Grok API request/response structures (native xAI SDK compatibility).

This is the merged 'best of both worlds' version of grok.py:
- Maintains native SDK style (`input` key, `to_sdk_messages()`, `from_dict`, `with_updates`)
- Includes rich professional documentation, full multimodal validation, and response parsing
- Retains Pyrefly compatibility (`__all__`, `GrokStreamingChunk`)
- Adds complete `GrokResponse` and `GrokBatchResponse` classes with native reasoning trace extraction

All classes are immutable dataclasses for thread-safety and predictability.

Classes are layered as follows:

- `GrokMessage`: Single message (text or multimodal).
- `GrokInput`: Sequence of messages (native 'input' wrapper).
- `GrokRequest`: Full request (model parameters, tools, reasoning, caching).
- `GrokBatchRequest`: Container for batch submission.
- `GrokResponse`: Parsed single response with reasoning trace.
- `GrokBatchResponse`: Parsed batch result.
- `GrokStreamingChunk`: Native streaming chunk (implements common protocol).

Flow of request use (single or streaming):
1. Create `GrokMessage` instances (or use `GrokMessage.from_dict`).
2. Combine into `GrokInput` (or use `GrokInput.from_list`).
3. Pass to `GrokRequest` with parameters (including `include_reasoning` and `reasoning_effort`).
4. Use `to_sdk_messages()` for the xAI SDK or pass the request object directly.

Flow of batch use:
1. Build one or more `GrokRequest` objects.
2. Pass to `GrokBatchRequest`.
3. Use `to_sdk_batch_requests()` for the SDK batch interface.

Flow of streaming:
1. Pass `stream=True` to the client.
2. Iterate over yielded `GrokStreamingChunk` objects.

Examples
--------
Basic text request with reasoning:
    >>> messages = [{"role": "user", "content": "What is 101*3?"}]
    >>> grok_input = GrokInput.from_list(messages)
    >>> request = GrokRequest(
    ...     model="grok-3-mini",
    ...     input=grok_input,
    ...     include_reasoning=True,
    ...     reasoning_effort="medium",
    ... )
    >>> sdk_messages = request.to_sdk_messages()

Batch of three requests:
    >>> req1 = GrokRequest(
    ...     model="grok-3-mini",
    ...     input=GrokInput.from_list([{"role": "user", "content": "Hello"}]),
    ... )
    >>> req2 = GrokRequest(
    ...     model="grok-3-mini",
    ...     input=GrokInput.from_list([{"role": "user", "content": "Weather in Sydney?"}]),
    ... )
    >>> req3 = GrokRequest(
    ...     model="grok-3-mini",
    ...     input=GrokInput.from_list([{"role": "user", "content": "Explain Rust lifetimes"}]),
    ... )
    >>> batch = GrokBatchRequest(batch_name="demo-batch", requests=[req1, req2, req3])

Streaming chunk example:
    >>> async for chunk in await client.generate(request, stream=True):
    ...     print(chunk.text, end="", flush=True)
    ...     if chunk.is_final:
    ...         print(f"\\nFinished: {chunk.finish_reason}")
"""

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, Sequence, cast, runtime_checkable

__all__ = [
    "GrokMessage",
    "GrokInput",
    "GrokRequest",
    "GrokBatchRequest",
    "GrokResponse",
    "GrokBatchResponse",
    "GrokStreamingChunk",
    "LLMStreamingChunkProtocol",
    "SaveMode",
    "Role",
]

type SaveMode = Literal["none", "json_files", "postgres"]
type Role = Literal["system", "user", "assistant", "developer"]


@runtime_checkable
class LLMStreamingChunkProtocol(Protocol):
    """Common contract for streaming chunks across providers."""

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


@dataclass(frozen=True)
class GrokMessage:
    """Immutable message supporting text, images, and file attachments.

    Parameters
    ----------
    role : Role
        Message role.
    content : str | list[dict[str, Any]]
        Text string or multimodal content list.
    """

    role: Role
    content: str | list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert the message to a dictionary for JSON serialisation.

        Returns
        -------
        dict[str, Any]
            Serialisable representation.
        """
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, msg_dict: dict[str, Any]) -> "GrokMessage":
        """Create GrokMessage from dict with full validation.

        Parameters
        ----------
        msg_dict : dict[str, Any]
            Dictionary with 'role' and 'content'.

        Returns
        -------
        GrokMessage
            Validated instance.

        Raises
        ------
        ValueError
            If keys missing, role invalid or content malformed.
        """
        cls._validate_keys(msg_dict)                                                      # early key check
        role = cls._validate_role(msg_dict["role"])
        content = msg_dict["content"]
        cls._validate_content(content)
        return cls(role=role, content=content)

    @staticmethod
    def _validate_keys(msg_dict: dict[str, Any]) -> None:
        """Validate required keys exist."""
        required_keys = {"role", "content"}
        if not required_keys.issubset(msg_dict.keys()):
            missing = required_keys - set(msg_dict.keys())
            raise ValueError(f"Missing keys: {missing}")

    @staticmethod
    def _validate_role(role_str: str) -> Role:
        """Validate and cast role string."""
        allowed_roles = ("system", "user", "assistant", "developer")
        if role_str not in allowed_roles:
            raise ValueError(
                f"Invalid role '{role_str}'. Must be one of: {allowed_roles}"
            )
        return cast(Role, role_str)

    @staticmethod
    def _validate_content(content: Any) -> None:
        """Validate content type and structure."""
        if isinstance(content, str):
            return
        if not isinstance(content, list):
            raise ValueError("Content must be str or list[dict].")
        for item in content:
            GrokMessage._validate_content_item(item)

    @staticmethod
    def _validate_content_item(item: dict[str, Any]) -> None:
        """Validate a single multimodal item."""
        if not isinstance(item, dict):
            raise ValueError("List items in content must be dicts.")
        typ = item.get("type")
        if typ == "input_text":
            if "text" not in item or not isinstance(item["text"], str):
                raise ValueError("input_text requires 'text': str.")
        elif typ == "input_image":
            if "image_url" not in item or not isinstance(item["image_url"], str):
                raise ValueError("input_image requires 'image_url': str.")
            detail = item.get("detail", "auto")
            if detail not in ("auto", "low", "high"):
                raise ValueError("detail must be 'auto', 'low', or 'high'.")
        elif typ is None:
            raise ValueError("Each item must have 'type' key.")
        else:
            raise ValueError(f"Invalid type '{typ}' in content item.")


@dataclass(frozen=True)
class GrokInput:
    """Wrapper for a sequence of GrokMessage instances (native 'input')."""

    messages: tuple[GrokMessage, ...] = field(default_factory=tuple)

    def to_list(self) -> list[dict[str, Any]]:
        """Native list-of-dict representation for xAI SDK.

        Returns
        -------
        list[dict[str, Any]]
            List of message dictionaries.
        """
        return [msg.to_dict() for msg in self.messages]

    @classmethod
    def from_str(cls, text: str) -> "GrokInput":
        """Convenience factory for single-user-message string input.

        Parameters
        ----------
        text : str
            User message text.

        Returns
        -------
        GrokInput
            Instance containing one user message.
        """
        return cls(messages=(GrokMessage(role="user", content=text),))

    @classmethod
    def from_list(
        cls, messages: Sequence[dict[str, Any] | GrokMessage] | None = None
    ) -> "GrokInput":
        """Create GrokInput from sequence of dicts or GrokMessage objects.

        Parameters
        ----------
        messages : Sequence[dict[str, Any] | GrokMessage] | None
            Input messages.

        Returns
        -------
        GrokInput
            Validated input wrapper.
        """
        if not messages:
            return cls(messages=tuple())
        processed: list[GrokMessage] = []
        for msg in messages:
            if isinstance(msg, GrokMessage):
                processed.append(msg)
            elif isinstance(msg, dict):
                processed.append(GrokMessage.from_dict(msg))
            else:
                raise TypeError(
                    f"Expected dict or GrokMessage, got {type(msg).__name__}"
                )
        return cls(messages=tuple(processed))


@dataclass(frozen=True)
class GrokRequest:
    """Full native Grok request with multimodal, caching, and batch support."""

    model: str
    input: GrokInput = field(default_factory=GrokInput)
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    include_reasoning: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None
    save_mode: SaveMode = "none"
    prompt_cache_key: str | None = None
    batch_request_id: str | None = None

    def to_sdk_messages(self) -> list[dict[str, Any]]:
        """Native representation for xAI SDK chat.create / append.

        Normalises legacy str input to GrokInput (single user message)
        for backward compatibility with tests and direct construction.
        """
        if isinstance(self.input, str):                                                   # legacy convenience path
            return GrokInput.from_str(self.input).to_list()
        if isinstance(self.input, GrokInput):
            return self.input.to_list()
        # fallback for list-of-dict (rare)
        return self.input if isinstance(self.input, list) else []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GrokRequest":
        """Convert generic input dict (may contain 'messages' or 'input') into typed native GrokRequest."""
        data = dict(data)
        if "messages" in data and "input" not in data:
            data["input"] = data.pop("messages")
        if isinstance(data.get("input"), (list, tuple)):
            data["input"] = GrokInput.from_list(data["input"])
        elif isinstance(data.get("input"), dict):
            data["input"] = GrokInput.from_list(data["input"].get("messages", []))
        return cls(**data)

    def get_messages(self) -> list[dict[str, Any]]:
        """Return normalised list of message dicts (role/content) for any input type.

        Reuses the existing to_sdk_messages() normalisation to guarantee consistency.
        """
        if isinstance(self.input, str):
            return [{"role": "user", "content": self.input}]
        # Complex input (object with .messages, list of dicts, etc.)
        messages = list(self.to_sdk_messages())
        return messages

    def to_chat_create_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for xAI SDK `chat.create()`.

        Centralises forwarding of all request parameters (tools, sampling
        controls, etc.) for efficiency and single source of truth.

        Returns
        -------
        dict[str, Any]
            Kwargs compatible with `AsyncClient.chat.create`.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "store_messages": True,                                                       # Enables Responses API + prompt caching
        }
        if self.tools is not None:
            kwargs["tools"] = self.tools
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format
            # reasoning_effort is model-dependent; handled by caller check
        return kwargs

    def extract_prompt_snippet(self, max_chars: int = 100) -> str:
        """Extract a truncated prompt snippet from the first user message.

        Handles both simple str input and complex message structures.
        """
        messages = self.get_messages()
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:max_chars]
                elif isinstance(content, list):
                    # For multimodal, join text parts only
                    text_parts: list[str] = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif (
                            isinstance(part, dict) and part.get("type") == "input_text"
                        ):
                            text_parts.append(part.get("text", ""))
                    snippet = " ".join(text_parts)
                    return snippet[:max_chars]
        return ""                                                                         # fallback for empty / system-only prompts

    def has_media(self) -> bool:
        """Return True if any user message contains multimodal content (images/files).

        Detects list-style content or dicts with image/file types.
        """
        messages = self.get_messages()
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype in ("input_image", "input_file"):
                            return True
                        # Future: extend for other media indicators if SDK evolves
        return False

    def with_updates(self, **updates: Any) -> "GrokRequest":
        """Type-safe replacement for object.__setattr__ (used by tests)."""
        return replace(self, **updates)


@dataclass(frozen=True)
class GrokBatchRequest:
    """Container for a batch of GrokRequests (used by client.batch.add)."""

    batch_name: str
    requests: list[GrokRequest] = field(default_factory=list)

    def to_sdk_batch_requests(self) -> list[Any]:
        """Will be converted by client to xAI SDK prepared objects."""
        return self.requests


@dataclass(frozen=True)
class GrokResponse:
    """Representation of a response from the xAI Grok API.

    Includes native reasoning trace extraction for both grok-3-mini and grok-4 families.
    """

    id: str
    created_at: int
    model: str
    output: Sequence[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    status: Literal["completed", "in_progress", "incomplete"] | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    reasoning_text: str | None = None

    @property
    def text(self) -> str:
        """Extract the main text content from the response."""
        parts: list[str] = []
        for item in self.output:
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    parts.append(part.get("text", ""))
        return "".join(parts).strip()

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Extract any tool calls from the response."""
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
        """Create a GrokResponse instance from a raw API response dictionary.

        Also extracts the native reasoning trace.
        """
        raw = data.copy()
        try:
            response_id = str(data["id"])
            created_at = int(data["created_at"])
            model = str(data["model"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Missing or invalid required field(s) in API response: "
                "'id', 'created_at', 'model'"
            ) from exc

        output = data.get("output", [])
        if not isinstance(output, list):
            output = []

        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            usage = None

        status = data.get("status")
        if status not in ("completed", "in_progress", "incomplete", None):
            status = None

            # Extract reasoning trace (model-dependent)
        reasoning_text: str | None = None
        for item in data.get("output", []):
            if item.get("type") == "message":
                reasoning_text = item.get("reasoning_content")
                break
            if isinstance(item.get("reasoning"), dict):
                reasoning_text = item.get("reasoning", {}).get("encrypted_content")
                break

        return cls(
            id=response_id,
            created_at=created_at,
            model=model,
            output=output,
            usage=usage,
            status=status,
            raw=raw,
            reasoning_text=reasoning_text,
        )

    def __str__(self) -> str:
        txt = self.text[:80] + "…" if len(self.text) > 80 else self.text
        tc = len(self.tool_calls)
        return f"GrokResponse(id={self.id}, model={self.model}, text={txt!r}, tool_calls={tc})"

    def get_messages(self) -> list[dict[str, Any]]:
        """Return the assistant message(s) from the Grok response in normalised format.

        Mirrors GrokRequest.get_messages() for consistent API usage across request/response objects.
        """
        messages: list[dict[str, Any]] = []
        try:
            # xAI SDK response follows OpenAI-compatible structure
            raw = getattr(
                self, "raw", getattr(self, "response", getattr(self, "data", {}))
            )
            choices = raw.get("choices", []) or raw.get("output", {}).get("choices", [])
            for choice in choices:
                message = choice.get("message") or choice.get("delta", {})
                if message:
                    role = message.get("role", "assistant")
                    content = message.get("content")
                    messages.append({"role": role, "content": content})

                    # Include reasoning/thinking content when present (for supported models)
                    reasoning = message.get("reasoning_content") or message.get(
                        "reasoning"
                    )
                    if reasoning:
                        messages.append({"role": "reasoning", "content": reasoning})
        except (AttributeError, TypeError, KeyError):
            # Graceful fallback for unexpected or minimal response structures
            pass
        return messages

    def extract_response_snippet(self, max_chars: int = 200) -> str:
        """Extract a truncated snippet of the main assistant response text.

        Mirrors GrokRequest.extract_prompt_snippet() for consistent logging/persistence.
        """
        messages = self.get_messages()
        for msg in messages:
            if msg.get("role") in ("assistant", "reasoning"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:max_chars]
                # Final fallback: direct raw lookup (common in xAI responses)
        try:
            raw = getattr(
                self, "raw", getattr(self, "response", getattr(self, "data", {}))
            )
            content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(content, str):
                return content[:max_chars]
        except (IndexError, AttributeError, TypeError):
            pass
        return ""

    def get_reasoning_content(self, max_chars: int = 500) -> str | None:
        """Return reasoning/thinking content if generated (for models that support include_reasoning)."""
        messages = self.get_messages()
        for msg in messages:
            if msg.get("role") == "reasoning":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:max_chars]
        return None

    def has_media(self) -> bool:
        """Return True if the response contains generated media (images, files, etc.).

        Currently always False for Grok text completions; implemented for future-proofing
        when image-generation or file-output capabilities are added to the xAI API.
        """
        # Grok-4 currently returns text-only output. Extend this method when needed.
        return False


@dataclass(frozen=True)
class GrokBatchResponse:
    """Immutable representation of a completed Grok batch result."""

    batch_id: str
    results: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GrokBatchResponse":
        """Create GrokBatchResponse from the raw /v1/batches/{id} response."""
        return cls(
            batch_id=str(data["id"]),
            results=tuple(data.get("results", [])),
        )


@dataclass(frozen=True)
class GrokStreamingChunk:
    """Native streaming chunk (implements LLMStreamingChunkProtocol)."""

    text: str = ""
    finish_reason: str | None = None
    tool_calls_delta: list[dict[str, Any]] | None = None
    is_final: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
