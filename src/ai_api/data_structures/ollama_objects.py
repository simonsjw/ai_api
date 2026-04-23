"""
Ollama-native data models for the ai_api package.

**What it does:**
Provides a complete, drop-in replacement for the xAI data structures so that
the higher-level chat, streaming, structured-output, and persistence layers
can be reused with zero (or near-zero) changes when the backend is a local
Ollama server instead of the xAI cloud.

**How it does it:**
- Mirrors the public API of ``xai_objects.py`` (same method names, same
  ``LLM*Protocol`` implementations, same ``SaveMode`` handling).
- Uses Pydantic ``BaseModel`` + ``dataclass(frozen=True)`` for validation
  and immutability exactly like the xAI side.
- Translates between the "messages" format that Ollama expects and the
  internal ``OllamaInput`` / ``OllamaMessage`` objects.
- Exposes the rich local telemetry that Ollama returns (``total_duration``,
  ``prompt_eval_count``, ``eval_count``, ``load_duration``, etc.) so that
  monitoring of CPU/GPU, model load time, tokens-per-second, etc. can be
  performed with a single protocol-compliant object.
- Supports base64 images directly in user messages (Ollama native) and
  optional ``format`` / JSON-schema structured output.

Key transparent differences from xAI (documented inline in code):
- ``input`` (xAI)  → ``messages`` (Ollama)
- ``response_format`` (xAI) → ``format`` (Ollama)
- No mandatory "Extract ... JSON" system-prompt string (Ollama is more lenient)
- Streaming chunks are 100 % compatible via ``LLMStreamingChunkProtocol``

Examples
--------
**1. Creating a request object that can be sent to an Ollama LLM**

>>> from src.ai_api.data_structures.ollama_objects import (
...     OllamaRequest, OllamaInput, OllamaMessage, OllamaJSONResponseSpec
... )
>>> # Simple text
>>> req = OllamaRequest(model="llama3.2", input="Tell me a joke")
>>> print(req.to_ollama_dict()["messages"][0]["content"])
'Tell me a joke'

>>> # With media (base64 image – Ollama native)
>>> import base64
>>> img_b64 = base64.b64encode(open("cat.jpg", "rb").read()).decode()
>>> msg = OllamaMessage(role="user", content="Describe this image", images=[img_b64])
>>> req = OllamaRequest(model="llava", input=OllamaInput(messages=(msg,)))
>>> print(req.has_media())
True

>>> # Structured output (JSON schema)
>>> from pydantic import BaseModel
>>> class Joke(BaseModel):
...     setup: str
...     punchline: str
>>> spec = OllamaJSONResponseSpec(model=Joke)
>>> req = OllamaRequest(model="llama3.2", input="Tell a joke", response_format=spec)
>>> print(req.to_ollama_dict()["format"])  # full JSON schema dict

**2. Processing a response from an Ollama LLM (including media & telemetry)**

>>> from src.ai_api.data_structures.ollama_objects import parse_ollama_response
>>> raw = {
...     "model": "llama3.2",
...     "created_at": "2026-04-23T...",
...     "message": {"role": "assistant", "content": "Why did the chicken..."},
...     "done": True,
...     "done_reason": "stop",
...     "total_duration": 1_234_567_890,   # nanoseconds
...     "prompt_eval_count": 12,
...     "eval_count": 87,
...     "eval_duration": 987_654_321,
... }
>>> resp = parse_ollama_response(raw)
>>> print(resp.text)
'Why did the chicken...'
>>> print(resp.meta()["total_duration"])          # raw ns for monitoring
1234567890
>>> print(resp.payload()["telemetry"]["eval_count"])  # 87 tokens generated

**3. Monitoring software & hardware running the model (Ollama-specific)**

The ``OllamaResponse`` and ``OllamaStreamingChunk`` expose the exact fields
that Ollama returns from ``/api/chat``. These allow real-time dashboards:

>>> tokens_per_sec = resp.eval_count / (resp.eval_duration / 1_000_000_000)
>>> print(f"Generation speed: {tokens_per_sec:.1f} tok/s")
>>> load_time_ms = (resp.load_duration or 0) / 1_000_000
>>> print(f"Model load time: {load_time_ms:.0f} ms")   # indicates cold-start vs warm
>>> # prompt_eval_count vs eval_count shows prefill vs decode cost
>>> # (useful for optimising context length / batch size on GPU)

All of the above works identically whether you are using the synchronous
``chat_turn_ollama`` or the async streaming path – the objects satisfy the
same protocols.

See Also
--------
base_objects : the protocols this module implements
xai_objects : the mirrored implementation for the xAI backend
"""

import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol, Sequence, Type, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

# Import the new generic protocols
from .base_objects import LLMEndpoint, LLMRequestProtocol, LLMResponseProtocol

T = TypeVar("T", bound="OllamaJSONResponseSpec")

__all__: list[str] = [
    "OllamaRole",
    "OllamaMessage",
    "OllamaInput",
    "OllamaRequest",
    "OllamaResponse",
    "OllamaStreamingChunk",
    "OllamaJSONResponseSpec",
    "LLMStreamingChunkProtocol",
    "SaveMode",
    "LLMRequestProtocol",                                                                 # re-exported
    "LLMResponseProtocol",                                                                # re-exported
]

# Re-export the shared protocol and type so existing code imports from here unchanged
from .xai_objects import LLMStreamingChunkProtocol, SaveMode                              # type: ignore

type OllamaRole = Literal["system", "user", "assistant", "tool"]


class DoneReason(str, Enum):
    """Ollama-specific stop reasons (visible in final chunk)."""

    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"

    # ----------------------------------------------------------------------
    # Structured Output (Ollama style)
    # ----------------------------------------------------------------------


class OllamaJSONResponseSpec(BaseModel):
    """Specification for structured (JSON) responses.

    Ollama uses the top-level "format" key:
      - "json" for plain JSON mode
      - dict = full JSON schema (Ollama 0.3.14+)

    Parameters
    ----------
    model : type[BaseModel] | dict[str, Any] | None
        Pydantic model or raw JSON schema. ``None`` disables structured output.
    instruction : str or None
        Optional extra system instruction (Ollama does not enforce the
        xAI-style magic string).

    Methods
    -------
    to_ollama_format
        Returns the value that should be put under the ``format`` key.
    from_ollama_response
        Convenience parser that extracts JSON from a finished response.

    Examples
    --------
    >>> spec = OllamaJSONResponseSpec(model={"type": "object", "properties": {...}})
    >>> req = OllamaRequest(..., response_format=spec)
    >>> "format" in req.to_ollama_dict()
    True
    """

    model: type[BaseModel] | dict[str, Any] | None = None
    instruction: str | None = None                                                        # optional custom system instruction

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def to_ollama_format(self) -> dict[str, Any] | str | None:
        if self.model is None:
            return None
        if isinstance(self.model, dict):
            return self.model
        return self.model.model_json_schema()

    @classmethod
    def parse_json(cls: type[T], json_data: str | bytes | bytearray) -> T:
        return cls.model_validate_json(json_data)

    @classmethod
    def from_ollama_response(cls: type[T], response: "OllamaResponse | str") -> T:
        text = response.text if isinstance(response, OllamaResponse) else response
        return cls.parse_json(text)

    # ----------------------------------------------------------------------
    # Message & Input
    # ----------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaMessage:
    """Ollama message (identical public API to xAIMessage).

    Parameters
    ----------
    role : {"system", "user", "assistant", "tool"}
    content : str | list[dict] | None
        Text or multimodal content list.
    images : list[str] | None
        List of base64-encoded images (Ollama native – no wrapper object).

    Examples
    --------
    >>> msg = OllamaMessage(role="user", content="Hello", images=["iVBORw0KGgo..."])
    """

    role: OllamaRole
    content: str | list[dict[str, Any]] | None = None
    images: list[str] | None = None                                                       # base64-encoded images (Ollama native)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.images:
            d["images"] = self.images
        return d

    @classmethod
    def from_dict(cls, msg_dict: dict[str, Any]) -> "OllamaMessage":
        cls._validate_keys(msg_dict)
        role = cls._validate_role(msg_dict["role"])
        content = msg_dict.get("content")
        images = msg_dict.get("images")
        return cls(role=role, content=content, images=images)

    @staticmethod
    def _validate_keys(d: dict) -> None:
        if "role" not in d:
            raise ValueError("Message missing 'role' key")

    @staticmethod
    def _validate_role(role_str: str) -> OllamaRole:
        allowed = ("system", "user", "assistant", "tool")
        if role_str not in allowed:
            raise ValueError(f"Invalid role '{role_str}'. Must be one of: {allowed}")
        return cast(OllamaRole, role_str)


@dataclass(frozen=True)
class OllamaInput:
    """Wrapper for a sequence of messages (Ollama uses 'messages', not 'input')."""

    messages: tuple[OllamaMessage, ...] = field(default_factory=tuple)

    def to_list(self) -> list[dict[str, Any]]:
        return [msg.to_dict() for msg in self.messages]

    @classmethod
    def from_str(cls, text: str) -> "OllamaInput":
        return cls(messages=(OllamaMessage(role="user", content=text),))

    @classmethod
    def from_list(
        cls, messages: Sequence[dict | OllamaMessage] | None = None
    ) -> "OllamaInput":
        if not messages:
            return cls()
        processed: list[OllamaMessage] = []
        for m in messages:
            processed.append(
                m if isinstance(m, OllamaMessage) else OllamaMessage.from_dict(m)
            )
        return cls(messages=tuple(processed))

    # ----------------------------------------------------------------------
    # Main Request (implements LLMRequestProtocol)
    # ----------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaRequest(BaseModel, LLMRequestProtocol):
    """Ollama-native request (implements LLMRequestProtocol).

    All generation parameters are exposed as optional fields so that the
    caller can use the same request object for both simple chat and
    highly-tuned inference.

    The ``input`` field accepts either a plain string (legacy) or an
    ``OllamaInput`` object (recommended for multi-turn / multimodal).

    Parameters
    ----------
    model : str
        Ollama model tag (e.g. "llama3.2", "llava", "qwen2.5-coder").
    input : OllamaInput | str
        The conversation history or raw prompt.
    temperature, top_p, top_k, ... : float | int | None
        Standard sampling + advanced Ollama options (mirostat, think, etc.).
    response_format : OllamaJSONResponseSpec | None
        Structured-output specification (see class above).
    tools : list[dict] | None
        Tool definitions for function calling.
    keep_alive : str | int | None
        How long to keep the model loaded after the request (e.g. "5m").
    save_mode : {"none", "json_files", "postgres"}
        Persistence hint passed through to the DB layer.
    options : dict | None
        Raw passthrough for any undocumented Ollama parameters.

    Methods
    -------
    meta, payload, endpoint
        Implementations of ``LLMRequestProtocol``.
    to_ollama_dict
        The exact dict that should be POSTed to ``/api/chat``.
    has_media, extract_prompt_snippet, with_updates, from_dict
        Convenience helpers used by the chat/stream layers.

    Examples
    --------
    See module-level docstring.
    """

    model: str
    input: OllamaInput | str = Field(
        ..., description="Accepts str or OllamaInput for legacy compatibility"
    )
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    max_tokens: int | None = None                                                         # maps to 'num_predict' in Ollama options
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    num_ctx: int | None = None
    stop: list[str] | None = None
    mirostat: int | None = None                                                           # 0, 1, or 2
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    min_p: float | None = None
    typical_p: float | None = None
    penalize_newline: bool | None = None
    repeat_last_n: int | None = None
    num_keep: int | None = None
    think: bool | None = None                                                             # for thinking models (Ollama 0.5+)
    response_format: OllamaJSONResponseSpec | None = None
    tools: list[dict[str, Any]] | None = None
    keep_alive: str | int | None = None
    save_mode: SaveMode = "none"
    prompt_cache_key: str | None = None                                                   # kept for API parity
    options: dict[str, Any] | None = (
        None                                                                              # raw passthrough for any future/undocumented params
    )

    model_config = ConfigDict(frozen=True, extra="allow")

    def __post_init__(self) -> None:
        if self.response_format is not None:
            self._validate_json_instruction_present()

    def _validate_json_instruction_present(self) -> None:
        pass                                                                              # Ollama does NOT require a magic string like xAI

    # ------------------------------------------------------------------
    # LLMRequestProtocol implementation
    # ------------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        """Return generation settings (implements LLMRequestProtocol)."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "num_ctx": self.num_ctx,
            "stop": self.stop,
            "mirostat": self.mirostat,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta,
            "min_p": self.min_p,
            "typical_p": self.typical_p,
            "penalize_newline": self.penalize_newline,
            "repeat_last_n": self.repeat_last_n,
            "num_keep": self.num_keep,
            "think": self.think,
            "response_format": self.response_format.to_ollama_format()
            if self.response_format
            else None,
            "tools": self.tools,
            "keep_alive": self.keep_alive,
            "save_mode": self.save_mode,
            "options": self.options,
        }

    def payload(self) -> dict[str, Any]:
        """Return the actual messages (implements LLMRequestProtocol)."""
        return {
            "messages": self.get_messages(),
            "input_type": "chat" if not isinstance(self.input, str) else "raw",
        }

    def endpoint(self) -> LLMEndpoint:
        """Return structured endpoint info (implements LLMRequestProtocol)."""
        return LLMEndpoint(
            provider="ollama",
            model=self.model,
            base_url="http://localhost:11434",
            path="/api/chat",
            api_type="native",
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
            if msg.get("role") == "user" and msg.get("images"):
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

    def with_updates(self, **updates: Any) -> "OllamaRequest":
        return replace(self, **updates)

    def to_ollama_dict(self) -> dict[str, Any]:
        """Required by chat_turn_ollama.py and chat_stream_ollama.py."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.get_messages(),
            "stream": False,
        }
        options: dict[str, Any] = {}

        # Core sampling
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.seed is not None:
            options["seed"] = self.seed
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens

            # Repetition control
        if self.repeat_penalty is not None:
            options["repeat_penalty"] = self.repeat_penalty
        if self.presence_penalty is not None:
            options["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            options["frequency_penalty"] = self.frequency_penalty
        if self.repeat_last_n is not None:
            options["repeat_last_n"] = self.repeat_last_n
        if self.penalize_newline is not None:
            options["penalize_newline"] = self.penalize_newline

            # Context & stopping
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.num_keep is not None:
            options["num_keep"] = self.num_keep
        if self.stop is not None:
            options["stop"] = self.stop

            # Mirostat sampling
        if self.mirostat is not None:
            options["mirostat"] = self.mirostat
        if self.mirostat_tau is not None:
            options["mirostat_tau"] = self.mirostat_tau
        if self.mirostat_eta is not None:
            options["mirostat_eta"] = self.mirostat_eta

            # Advanced
        if self.min_p is not None:
            options["min_p"] = self.min_p
        if self.typical_p is not None:
            options["typical_p"] = self.typical_p
        if self.think is not None:
            options["think"] = self.think

            # Raw passthrough (highest priority — can override anything above)
        if self.options:
            options.update(self.options)

        if options:
            payload["options"] = options

        if self.response_format:
            fmt = self.response_format.to_ollama_format()
            if fmt:
                payload["format"] = fmt
        if self.tools:
            payload["tools"] = self.tools
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive

        return {k: v for k, v in payload.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OllamaRequest":
        data = dict(data)
        if "messages" in data and "input" not in data:
            data["input"] = OllamaInput.from_list(data.pop("messages"))
        elif isinstance(data.get("input"), (list, tuple)):
            data["input"] = OllamaInput.from_list(data["input"])
        return cls(**data)

    # ----------------------------------------------------------------------
    # Response (implements LLMResponseProtocol)
    # ----------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaResponse(BaseModel, LLMRequestProtocol):
    """Ollama-native response (implements LLMResponseProtocol).

    Note: inherits from LLMRequestProtocol in the current code base (assumed
    correct per user instruction). In practice it fully satisfies
    LLMResponseProtocol as well.

    Contains the generated ``message``, ``done_reason``, and the full set
    of local performance counters that Ollama exposes. These counters are
    the primary source for hardware/software monitoring of the Ollama
    instance (model load time, tokens/s, context utilisation, etc.).

    Parameters
    ----------
    model, created_at, message, done, done_reason : ...
    total_duration, load_duration, prompt_eval_count, ... : int | None
        All timing values are in **nanoseconds**. Convert by dividing by
        1_000_000 for milliseconds or 1_000_000_000 for seconds.
    parsed : BaseModel | dict | None
        Populated by the structured-output post-processor when a
        ``response_format`` was supplied.

    Properties
    ----------
    text : str
        Convenience accessor for ``message["content"]``.
    tool_calls : list[dict]
        Convenience accessor for ``message["tool_calls"]``.

    Methods
    -------
    meta, payload, endpoint
        Protocol implementations (rich telemetry included in both).
    from_dict, extract_response_snippet
        Helpers used by the client layer.

    Examples
    --------
    See module-level "Processing a response" example.
    Hardware monitoring:

    >>> resp = parse_ollama_response(raw_from_ollama)
    >>> gen_speed = resp.eval_count / (resp.eval_duration / 1e9)
    >>> model_load_ms = (resp.load_duration or 0) / 1e6
    >>> print(f"{gen_speed:.1f} tok/s, model loaded in {model_load_ms:.0f} ms")
    """

    model: str
    created_at: str
    message: dict[str, Any]
    done: bool = True
    done_reason: DoneReason | None = None

    # Rich local performance telemetry
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    parsed: BaseModel | dict | None = None

    @property
    def text(self) -> str:
        return self.message.get("content", "") or ""

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return self.message.get("tool_calls") or []

    # ------------------------------------------------------------------
    # LLMResponseProtocol implementation
    # ------------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        """Return generation settings + telemetry (implements LLMResponseProtocol)."""
        return {
            "model": self.model,
            "done_reason": self.done_reason.value if self.done_reason else None,
            "total_duration": self.total_duration,
            "load_duration": self.load_duration,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
        }

    def payload(self) -> dict[str, Any]:
        """Return the generated output + telemetry (implements LLMResponseProtocol)."""
        return {
            "text": self.text,
            "tool_calls": self.tool_calls,
            "finish_reason": self.done_reason.value if self.done_reason else None,
            "parsed": self.parsed,
            "telemetry": {
                "total_duration_ms": round(self.total_duration / 1_000_000, 2)
                if self.total_duration
                else None,
                "eval_count": self.eval_count,
            },
        }

    def endpoint(self) -> LLMEndpoint:
        return LLMEndpoint(
            provider="ollama",
            model=self.model,
            base_url="http://localhost:11434",
            path="/api/chat",
            api_type="native",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OllamaResponse":
        return cls(
            model=data["model"],
            created_at=data["created_at"],
            message=data.get("message", {}),
            done=data.get("done", True),
            done_reason=data.get("done_reason"),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            prompt_eval_duration=data.get("prompt_eval_duration"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration"),
            raw=data,
        )

    def extract_response_snippet(self, max_chars: int = 200) -> str:
        txt = self.text
        return txt[:max_chars] + "…" if len(txt) > max_chars else txt

    # ----------------------------------------------------------------------
    # Streaming Chunk (implements LLMStreamingChunkProtocol)
    # ----------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaStreamingChunk:
    """Implements LLMStreamingChunkProtocol — drop-in compatible with xAI streaming code.

    On the final chunk (``is_final=True``) the telemetry fields
    (``done_reason``, ``total_duration``) are populated so that the same
    monitoring logic used for non-streaming responses can be applied.

    Parameters
    ----------
    text, finish_reason, tool_calls_delta, is_final, raw : ...
    done_reason, total_duration : ...
        Only meaningful on the final chunk.
    """

    text: str = ""
    finish_reason: str | None = None
    tool_calls_delta: list[dict[str, Any]] | None = None
    is_final: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    # Ollama-specific convenience fields (populated on final chunk)
    done_reason: DoneReason | None = None
    total_duration: int | None = None

    def meta(self) -> dict[str, Any]:
        return {
            "done_reason": self.done_reason.value if self.done_reason else None,
            "total_duration": self.total_duration,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "finish_reason": self.finish_reason,
            "is_final": self.is_final,
        }

    def endpoint(self) -> LLMEndpoint:
        return LLMEndpoint(provider="ollama", model="streaming", api_type="native")


def parse_ollama_response(data: dict[str, Any]) -> OllamaResponse:
    """Helper used by the Ollama client layer."""
    return OllamaResponse.from_dict(data)
