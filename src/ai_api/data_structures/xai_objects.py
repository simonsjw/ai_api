"""
xAI SDK data models for the ai_api package.

This module defines clean, type-safe dataclasses and Pydantic models that map
directly to the official xAI Python SDK (`xai_sdk`). These structures serve as
the canonical abstraction layer for constructing requests, handling responses,
and configuring features such as structured JSON output, multimodal content,
and streaming.

All public classes and helpers translate seamlessly into the parameters and
objects expected by `xai_sdk.AsyncClient.chat.create()` (and related methods)
via dedicated conversion methods (e.g., `to_api_kwargs()`, `to_sdk_response_format()`).

Public exports (via ``__all__``) include:

- **Request models**: ``xAIMessage``, ``xAIInput``, ``xAIRequest``
- **Structured output**: ``xAIJSONResponseSpec``
- **Batch support**: ``xAIBatchRequest``, ``xAIBatchResponse``
- **Streaming & protocol**: ``xAIStreamingChunk``, ``LLMStreamingChunkProtocol``
- **Supporting types**: ``SaveMode``, ``Role``

Design principles:
- Full compatibility with the xAI Responses API / OpenAI-compatible endpoint.
- Direct support for the official SDK client patterns.
- Immutable, validated data structures with helper methods for safe usage.
- Extensible persistence configuration via ``SaveMode``.

Application code interacts primarily with ``xAIRequest`` and ``xAIResponse``
while the lower-level models ensure correct mapping to the xAI SDK.

Multimodal attachment (now the canonical way for turn/stream modes):
    content = [
        {"type": "input_text", "text": "Describe this"},
        {"type": "input_image", "image_url": "https://... or /local/path.jpg"},
        {"type": "input_file",  "file_url": "/local/path.pdf"}
    ]
"""

import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import (
    Any,
    Literal,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    cast,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field
from xai_sdk import AsyncClient as XAIAsyncClient
from xai_sdk.chat import assistant, system, user

# Import the new generic protocols
from .base_objects import LLMEndpoint, LLMRequestProtocol, LLMResponseProtocol

T = TypeVar("T", bound="xAIJSONResponseSpec")

__all__: list[str] = [
    "xAIMessage",
    "xAIInput",
    "xAIRequest",
    "xAIBatchRequest",
    "xAIResponse",
    "xAIBatchResponse",
    "xAIStreamingChunk",
    "LLMStreamingChunkProtocol",
    "SaveMode",
    "Role",
    "xAIJSONResponseSpec",
    "JSON_INSTRUCTION",
    "LLMRequestProtocol",                                                                 # re-exported
    "LLMResponseProtocol",                                                                # re-exported
]

type SaveMode = Literal["none", "json_files", "postgres"]
type Role = Literal["system", "user", "assistant", "developer"]

# Required system-prompt substring that xAI expects when JSON mode is active.
# The check is performed on the full content string (case-sensitive).
JSON_INSTRUCTION: str = "Extract the requested information as structured JSON."


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

    # ----------------------------------------------------------------------
    # Structured Output
    # ----------------------------------------------------------------------


@dataclass(frozen=True)
class xAIJSONResponseSpec(BaseModel):
    """Specification for enforcing JSON-structured responses.

    Accepts either a Pydantic ``BaseModel`` subclass (recommended) or a raw
    JSON-schema dictionary. When attached to an ``xAIRequest``, the underlying
    xAI SDK automatically sets the appropriate ``response_format`` and the
    request validator ensures the system prompt contains the required
    instruction.

    Parameters
    ----------
    model : type[BaseModel] | dict[str, Any] | None
        Pydantic model class or raw JSON schema. ``None`` disables structured
        output.
    """

    model: type[BaseModel] | dict[str, Any] | None = None

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def to_sdk_response_format(self) -> dict[str, Any] | None:
        """Convert to the exact format expected by the xAI Responses API / OpenAI-compatible endpoint."""
        if self.model is None:
            return None

        if isinstance(self.model, dict):
            schema = self.model
            name = "structured_output"
        else:
            schema = self.model.model_json_schema()
            name = getattr(self.model, "__name__", "structured_output")

        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
                "strict": True,
            },
        }

    @classmethod
    def parse_json(cls: type[T], json_data: str | bytes | bytearray) -> T:
        return cls.model_validate_json(json_data)

    @classmethod
    def from_xai_response(cls: type[T], response: "xAIResponse | str") -> T:
        if isinstance(response, str):
            json_data = response
        else:
            if not response.text:
                raise ValueError("xAIResponse contains no text content to parse")
            json_data = response.text
        return cls.parse_json(json_data)

    # ----------------------------------------------------------------------
    # Message & Input
    # ----------------------------------------------------------------------


@dataclass(frozen=True)
class xAIMessage:
    """Immutable message supporting text, images, and file attachments."""

    role: Role
    content: str | list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, msg_dict: dict[str, Any]) -> "xAIMessage":
        cls._validate_keys(msg_dict)
        role = cls._validate_role(msg_dict["role"])
        content = msg_dict["content"]
        cls._validate_content(content)
        return cls(role=role, content=content)

    @staticmethod
    def _validate_keys(msg_dict: dict[str, Any]) -> None:
        required_keys = {"role", "content"}
        if not required_keys.issubset(msg_dict.keys()):
            missing = required_keys - set(msg_dict.keys())
            raise ValueError(f"Missing keys: {missing}")

    @staticmethod
    def _validate_role(role_str: str) -> Role:
        allowed_roles = ("system", "user", "assistant", "developer")
        if role_str not in allowed_roles:
            raise ValueError(
                f"Invalid role '{role_str}'. Must be one of: {allowed_roles}"
            )
        return cast(Role, role_str)

    @staticmethod
    def _validate_content(content: Any) -> None:
        if isinstance(content, str):
            return
        if not isinstance(content, list):
            raise ValueError("Content must be str or list[dict].")
        for item in content:
            xAIMessage._validate_content_item(item)

    @staticmethod
    def _validate_content_item(item: dict[str, Any]) -> None:
        if not isinstance(item, dict):
            raise ValueError("Content item must be a dict.")
        if "type" not in item:
            raise ValueError("Content item missing 'type' key.")


@dataclass(frozen=True)
class xAIInput:
    """Wrapper for a sequence of messages (xAI uses 'input' internally)."""

    messages: tuple[xAIMessage, ...] = field(default_factory=tuple)

    def to_list(self) -> list[dict[str, Any]]:
        return [msg.to_dict() for msg in self.messages]

    @classmethod
    def from_str(cls, text: str) -> "xAIInput":
        return cls(messages=(xAIMessage(role="user", content=text),))

    @classmethod
    def from_list(
        cls, messages: Sequence[dict | xAIMessage] | None = None
    ) -> "xAIInput":
        if not messages:
            return cls()
        processed: list[xAIMessage] = []
        for m in messages:
            processed.append(
                m if isinstance(m, xAIMessage) else xAIMessage.from_dict(m)
            )
        return cls(messages=tuple(processed))

    # ----------------------------------------------------------------------
    # Main Request (implements LLMRequestProtocol)
    # ----------------------------------------------------------------------


@dataclass(frozen=True)
class xAIRequest(BaseModel, LLMRequestProtocol):
    """xAI-native request (implements LLMRequestProtocol)."""

    model: str
    input: xAIInput | str = Field(..., description="Accepts str or xAIInput")
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: xAIJSONResponseSpec | None = None
    tools: list[dict[str, Any]] | None = None
    save_mode: SaveMode = "none"
    base_url: str | None = None
    prompt_cache_key: str | None = None

    model_config = ConfigDict(frozen=True)

    # ------------------------------------------------------------------
    # LLMRequestProtocol implementation
    # ------------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        """Return generation settings (implements LLMRequestProtocol)."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": self.response_format.to_sdk_response_format()
            if self.response_format
            else None,
            "tools": self.tools,
            "save_mode": self.save_mode,
            "prompt_cache_key": self.prompt_cache_key,
        }

    def payload(self) -> dict[str, Any]:
        """Return the actual messages/prompt (implements LLMRequestProtocol)."""
        if isinstance(self.input, str):
            return {"prompt": self.input, "input_type": "raw"}
        return {"messages": self.input.to_list(), "input_type": "chat"}

    def endpoint(self) -> LLMEndpoint:
        """Return structured endpoint info (implements LLMRequestProtocol)."""
        return LLMEndpoint(
            provider="xai",
            model=self.model,
            base_url=self.base_url or "https://api.x.ai/v1",
            path="/chat/completions",
            api_type="sdk",
        )

    # ------------------------------------------------------------------
    # Existing helper methods (kept for backward compatibility)
    # ------------------------------------------------------------------
    def get_messages(self) -> list[dict[str, Any]]:
        if isinstance(self.input, str):
            return [{"role": "user", "content": self.input}]
        return self.input.to_list()

    def has_media(self) -> bool:
        for msg in self.get_messages():
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in (
                            "input_image",
                            "input_file",
                        ):
                            return True
        return False

    def extract_prompt_snippet(self, max_chars: int = 100) -> str:
        msgs = self.get_messages()
        for m in msgs:
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    return content[:max_chars]
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return " ".join(text_parts)[:max_chars]
        return ""

    def with_updates(self, **updates: Any) -> "xAIRequest":
        return replace(self, **updates)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "xAIRequest":
        data = dict(data)
        if "messages" in data and "input" not in data:
            data["input"] = xAIInput.from_list(data.pop("messages"))
        elif isinstance(data.get("input"), (list, tuple)):
            data["input"] = xAIInput.from_list(data["input"])
        return cls(**data)

    def to_sdk_chat_kwargs(self) -> dict[str, Any]:
        """Required by chat_turn_xai.py and xai_client.py."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.get_messages(),
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.response_format:
            kwargs["response_format"] = self.response_format.to_sdk_response_format()
        if self.tools:
            kwargs["tools"] = self.tools
        return kwargs

    def prepare_batch_chat(self) -> dict[str, Any]:
        """Required by chat_batch_xai.py."""
        return self.to_sdk_chat_kwargs()

    # ----------------------------------------------------------------------
    # Response (implements LLMResponseProtocol)
    # ----------------------------------------------------------------------


@dataclass(frozen=True)
class xAIResponse(BaseModel, LLMResponseProtocol):
    """xAI response wrapper (implements LLMResponseProtocol)."""

    model: str
    created_at: str | None = None
    choices: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    parsed: BaseModel | dict | None = None

    @property
    def text(self) -> str:
        if self.choices:
            return self.choices[0].get("message", {}).get("content", "") or ""
        return ""

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        if self.choices:
            return self.choices[0].get("message", {}).get("tool_calls", []) or []
        return []

    # ------------------------------------------------------------------
    # LLMResponseProtocol implementation
    # ------------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.choices[0].get("finish_reason")
            if self.choices
            else None,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": self.tool_calls,
            "finish_reason": self.choices[0].get("finish_reason")
            if self.choices
            else None,
            "parsed": self.parsed,
        }

    def endpoint(self) -> LLMEndpoint:
        return LLMEndpoint(
            provider="xai",
            model=self.model,
            base_url="https://api.x.ai/v1",
            path="/chat/completions",
            api_type="sdk",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "xAIResponse":
        return cls(
            model=data.get("model", "unknown"),
            created_at=data.get("created_at"),
            choices=data.get("choices", []),
            usage=data.get("usage"),
            raw=data,
        )

    @classmethod
    def from_sdk(cls, sdk_response: Any) -> "xAIResponse":
        """Required by chat_turn_xai.py."""
        return cls.from_dict(
            sdk_response.model_dump()
            if hasattr(sdk_response, "model_dump")
            else sdk_response
        )

    def set_parsed(self, parsed: BaseModel | dict) -> None:
        """Required by response_struct_xai.py."""
        object.__setattr__(self, "parsed", parsed)                                        # since frozen dataclass

        # ----------------------------------------------------------------------
        # Batch Support
        # ----------------------------------------------------------------------


@dataclass(frozen=True)
class xAIBatchRequest(BaseModel):
    requests: list[xAIRequest]

    def meta(self) -> dict[str, Any]:
        return {"batch_size": len(self.requests), "provider": "xai"}

    def payload(self) -> dict[str, Any]:
        return {"requests": [r.payload() for r in self.requests]}

    def endpoint(self) -> LLMEndpoint:
        return LLMEndpoint(provider="xai", model="batch", api_type="batch")


@dataclass(frozen=True)
class xAIBatchResponse(BaseModel):
    responses: list[xAIResponse]
    batch_id: str | None = None

    def meta(self) -> dict[str, Any]:
        return {"batch_size": len(self.responses), "provider": "xai"}

    def payload(self) -> dict[str, Any]:
        return {"responses": [r.payload() for r in self.responses]}

    def endpoint(self) -> LLMEndpoint:
        return LLMEndpoint(provider="xai", model="batch", api_type="batch")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "xAIBatchResponse":
        """Required by chat_batch_xai.py."""
        responses = [xAIResponse.from_dict(r) for r in data.get("responses", [])]
        return cls(responses=responses, batch_id=data.get("batch_id"))

    # ----------------------------------------------------------------------
    # Streaming Chunk
    # ----------------------------------------------------------------------


@dataclass(frozen=True)
class xAIStreamingChunk:
    """Native streaming chunk (implements LLMStreamingChunkProtocol)."""

    text: str = ""
    finish_reason: str | None = None
    tool_calls_delta: list[dict[str, Any]] | None = None
    is_final: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        obj_dict = self.__dict__.copy()
        obj_dict.pop("raw", None)
        self.raw.update(obj_dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "xAIStreamingChunk":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "finish_reason": self.finish_reason,
            "tool_calls_delta": self.tool_calls_delta,
            "is_final": self.is_final,
            "raw": self.raw,
        }

    def __str__(self) -> str:
        return f"xAIStreamingChunk(text={self.text!r}, finish_reason={self.finish_reason}, is_final={self.is_final})"

    def __repr__(self) -> str:
        return (
            f"xAIStreamingChunk(text={self.text!r}, finish_reason={self.finish_reason!r}, "
            f"tool_calls_delta={self.tool_calls_delta}, is_final={self.is_final})"
        )
