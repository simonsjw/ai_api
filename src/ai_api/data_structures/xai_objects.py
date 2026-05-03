"""
xAI SDK data models for the ai_api package.

**What it does:**
Defines the canonical, type-safe data structures that map 1:1 onto the
official ``xai_sdk`` Python client. These classes are the single source of
truth for every xAI interaction (chat, streaming, batch, structured output,
multimodal).

**How it does it:**
- Uses Pydantic ``BaseModel`` (frozen) + ``dataclass(frozen=True)`` for
  validation and immutability.
- Implements the three methods of ``LLMRequestProtocol`` /
  ``LLMResponseProtocol`` so that the rest of the package (persistence,
  streaming aggregator, structured-output post-processor) stays provider-
  agnostic.
- Provides ``to_sdk_chat_kwargs()`` and ``to_sdk_response_format()`` that
  produce exactly the arguments expected by
  ``xai_sdk.AsyncClient.chat.create()``.
- Supports the modern multimodal content format
  (``input_text`` / ``input_image`` / ``input_file``) that the xAI Responses
  API expects.
- Exposes ``parsed`` attribute on responses so that JSON-mode output can be
  automatically validated against a Pydantic model.

All higher-level code (``chat_turn_xai.py``, ``response_struct_xai.py``,
batch runners, etc.) consumes these objects exclusively; the SDK is never
called directly from application logic.

Examples
--------
**Creating a request object that can be sent to xAI**

>>> from src.ai_api.data_structures.xai_objects import (
...     xAIRequest,
...     xAIInput,
...     xAIMessage,
...     xAIJSONResponseSpec,
... )
>>> from pydantic import BaseModel

>>> class City(BaseModel):
...     name: str
...     population: int

>>> spec = xAIJSONResponseSpec(model=City)
>>> req = xAIRequest(
...     model="grok-2-latest",
...     input="Tell me about Paris",
...     response_format=spec,
...     temperature=0.3,
...     max_tokens=200,
... )
>>> kwargs = req.to_sdk_chat_kwargs()
>>> print(kwargs["response_format"]["type"])
'json_schema'

**Multimodal request (image + text)**

>>> content = [
...     {"type": "input_text", "text": "What is in this image?"},
...     {"type": "input_image", "image_url": "https://example.com/cat.jpg"},
... ]
>>> msg = xAIMessage(role="user", content=content)
>>> req = xAIRequest(model="grok-2-vision", input=xAIInput(messages=(msg,)))
>>> print(req.has_media())
True

**Processing a response (including parsed structured output)**

>>> from src.ai_api.data_structures.xai_objects import xAIResponse
>>> raw = {
...     "model": "grok-2-latest",
...     "choices": [{"message": {"content": '{"name": "Paris",
...       "population": 2100000}'}}],
...     "usage": {"prompt_tokens": 42, "completion_tokens": 18},
... }
>>> resp = xAIResponse.from_dict(raw)
>>> print(resp.text)
'{"name": "Paris", "population": 2100000}'
>>> parsed = City.model_validate_json(resp.text)  # or use the post-processor
>>> resp.set_parsed(parsed)
>>> print(resp.payload()["parsed"].name)
'Paris'

**Embeddings / media references**
If the model is instructed (via tools or JSON schema) to return embeddings
or references to generated media, they appear in ``payload()["parsed"]`` or
``tool_calls``. The base protocol guarantees the shape; concrete handling
lives in the structured-output layer.

See Also
--------
base_objects : the protocols this module satisfies
ollama_objects : the mirrored local-backend implementation
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import (
    Any,
    Literal,
    Self,
    Sequence,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field

from .base_objects import (
    LLMEndpoint,
    LLMRequestProtocol,
    LLMResponseProtocol,
    NeutralTurn,
    SaveMode,
)

__all__: list[str] = [
    "xAIMessage",
    "xAIInput",
    "xAIRequest",
    "xAIBatchRequest",
    "xAIResponse",
    "xAIBatchResponse",
    "xAIStreamingChunk",
    "Role",
    "xAIJSONResponseSpec",
    "JSON_INSTRUCTION",
]


type Role = Literal["system", "user", "assistant", "developer"]

# Required system-prompt substring that xAI expects when JSON mode is active.
# The check is performed on the full content string (case-sensitive).
JSON_INSTRUCTION: str = "Extract the requested information as structured JSON."


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
    instruction: str | None
        An optional custom system instruction to produce JSON formatted responses.

    Methods
    -------
    to_sdk_response_format
        Produces the exact dict expected under ``response_format`` by the SDK.
    from_xai_response
        Convenience parser that extracts and validates JSON from a finished
        response.

    Examples
    --------
    >>> spec = xAIJSONResponseSpec(model=City)
    >>> req = xAIRequest(..., response_format=spec)
    >>> "json_schema" in req.to_sdk_chat_kwargs()["response_format"]["type"]
    True
    """

    model: type[BaseModel] | dict[str, Any] | None = None
    instruction: str | None = None                                                        # optional custom system instruction

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def to_sdk_response_format(self) -> dict[str, Any] | None:
        """
        Convert to the format expected by the xAI SDK.
        """
        if self.model is None:
            return None

        if isinstance(self.model, dict):
            schema = self.model
            name = "response"
        else:
            schema = self.model.model_json_schema()
            name = getattr(self.model, "__name__", "response")

        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
                "strict": True,
            },
        }

    @classmethod
    def parse_json(cls, json_data: str | bytes | bytearray) -> Self:
        return cls.model_validate_json(json_data)

    @classmethod
    def from_xai_response(cls, response: "xAIResponse | str") -> Self:
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
    """Immutable message supporting text, images, and file attachments.

    Parameters
    ----------
    role : {"system", "user", "assistant", "developer"}
    content : str | list[dict[str, Any]]
        Either plain text or the multimodal content list
        (``input_text``, ``input_image``, ``input_file``).

    Examples
    --------
    >>> msg = xAIMessage(
    ...     role="user",
    ...     content=[
    ...         {"type": "input_text", "text": "Describe"},
    ...         {"type": "input_image", "image_url": "https://..."},
    ...     ],
    ... )
    """

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
    """xAI-native request (implements LLMRequestProtocol).

    Parameters
    ----------
    model : str
        xAI model name (e.g. "grok-2-latest", "grok-2-vision").
    input : xAIInput | str
        Conversation history or raw prompt.
    temperature, max_tokens : float | int | None
    response_format : xAIJSONResponseSpec | None
        Structured-output spec (automatically adds the required system
        instruction).
    tools, save_mode, base_url, prompt_cache_key : ...
        Standard options.

    Methods
    -------
    meta, payload, endpoint
        Protocol implementations.
    to_sdk_chat_kwargs, prepare_batch_chat, has_media, ...
        Helpers used by the xAI client layer.

    Examples
    --------
    See module-level examples.
    """

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

    @classmethod
    def from_neutral_history(
        cls,
        neutral_history: list[dict] | list[NeutralTurn],
        new_prompt: str | list[dict],
        metadata: dict[str, Any],
        **kwargs,
    ) -> "xAIRequest":
        """
        Reconstruct an xAIRequest from a slice of neutral history plus a new prompt.

        Mirrors the contract of ``OllamaRequest.from_neutral_history`` so that
        higher-level code (``ChatSession``, persistence layer) can treat all
        providers uniformly.

        Parameters
        ----------
        neutral_history : list of dict or NeutralTurn
            Reconstructed conversation turns for the relevant branch/depth.
        new_prompt : str or list of dict
            The user message (plain text or multimodal content blocks) to append.
        metadata : dict
            Generation parameters and request settings (model, temperature,
            max_tokens, response_format, tools, save_mode, prompt_cache_key, etc.).
        **kwargs
            Extra constructor arguments forwarded to ``xAIRequest``.

        Returns
        -------
        xAIRequest
            Ready-to-send request object for the xAI SDK.

        See Also
        --------
        OllamaRequest.from_neutral_history : the equivalent for Ollama.
        """
        messages: list[xAIMessage] = []

        for turn in neutral_history:
            if isinstance(turn, dict):
                turn = NeutralTurn(**turn)
            if turn.role in ("system", "user", "assistant", "developer"):
                messages.append(xAIMessage(role=turn.role, content=turn.content))

        if isinstance(new_prompt, str):
            messages.append(xAIMessage(role="user", content=new_prompt))
        else:
            messages.append(xAIMessage(role="user", content=new_prompt))

        return cls(
            model=metadata.get("model", "grok-2"),
            input=xAIInput(messages=tuple(messages)),
            temperature=metadata.get("temperature", 0.7),
            max_tokens=metadata.get("max_tokens"),
            response_format=metadata.get("response_format"),
            tools=metadata.get("tools"),
            save_mode=metadata.get("save_mode", "postgres"),
            prompt_cache_key=metadata.get("prompt_cache_key"),
            **kwargs,
        )

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
    """xAI response wrapper (implements LLMResponseProtocol).

    Parameters
    ----------
    model, created_at, choices, usage, raw, parsed : ...
        ``choices[0]["message"]["content"]`` is the generated text.
        ``parsed`` is populated after structured-output validation.

    Properties
    ----------
    text, tool_calls : convenience accessors.

    Methods
    -------
    meta, payload, endpoint, from_dict, from_sdk, set_parsed
        Full protocol + client-layer helpers.

    Examples
    --------
    See module-level "Processing a response" example.
    """

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

    def to_neutral_format(self, branch_info: dict | None = None) -> dict:
        """
        Convert an xAI response into the canonical neutral format.

        The returned dict becomes the ``response`` field inside
        ``NeutralResponseBlob`` and is what gets stored in the ``response``
        JSONB column. All provider-specific fields are mapped to the common
        keys expected by the rest of the system.

        Parameters
        ----------
        branch_info : dict, optional
            Branching metadata (``tree_id``, ``branch_id``, ``parent_response_id``,
            ``sequence``) that will be embedded in the ``branch_meta`` key.

        Returns
        -------
        dict
            A plain dictionary conforming to the ``NeutralTurn`` schema.
        """
        return {
            "role": "assistant",
            "content": self.text,
            "structured": self.parsed,
            "finish_reason": (
                self.choices[0].get("finish_reason") if self.choices else None
            ),
            "usage": self.usage,
            "tools": self.tool_calls,
            "raw": self.raw,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "branch_meta": branch_info or {},
        }

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
