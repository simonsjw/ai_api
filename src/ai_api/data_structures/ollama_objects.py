"""
Ollama-native data models for the ai_api package.

Mirrors the exact structure, validation style, and public API of xai_objects.py
so that the rest of the package (streaming, persistence, structured output, etc.)
can be reused with almost zero changes.

Key transparent differences (documented inline):
- xAI uses "input" (Responses API / SDK) → Ollama uses "messages" (standard chat format).
- Ollama returns rich local telemetry (total_duration, prompt_eval_count, etc.).
- Ollama supports base64 images directly in the message (no "input_image" wrapper).
- Structured output uses "format" (json or JSON schema) instead of xAI's response_format.
- Streaming chunks are identical via LLMStreamingChunkProtocol.

All factories and helpers are named consistently with the xAI side.
"""

import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol, Sequence, Type, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound="OllamaJSONResponseSpec")

__all__: list[str] = [
    "OllamaRole",
    "OllamaMessage",
    "OllamaInput",
    "OllamaRequest",
    "OllamaResponse",
    "OllamaStreamingChunk",
    "OllamaJSONResponseSpec",
    "LLMStreamingChunkProtocol",                                                          # re-exported for convenience
    "SaveMode",                                                                           # re-exported
]

# Re-export the shared protocol and type so existing code imports from here unchanged
from .xai_objects import LLMStreamingChunkProtocol, SaveMode                              # type: ignore

# ----------------------------------------------------------------------
# Enums & simple types (kept identical to xAI for drop-in compatibility)
# ----------------------------------------------------------------------
type OllamaRole = Literal["system", "user", "assistant", "tool"]


class DoneReason(str, Enum):
    """Ollama-specific stop reasons (visible in final chunk)."""

    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"

    # ----------------------------------------------------------------------
    # Structured output support (Ollama style)
    # ----------------------------------------------------------------------


class OllamaJSONResponseSpec(BaseModel):
    """Specification for structured (JSON) responses.

    Ollama uses the top-level "format" key:
      - "json" for plain JSON mode
      - dict = full JSON schema (Ollama 0.3.14+)
    """

    model: type[BaseModel] | dict[str, Any] | None = None
    instruction: str | None = None                                                        # optional custom system instruction

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def to_ollama_format(self) -> dict[str, Any] | str | None:
        """Return the exact value expected in the 'format' field of an Ollama request."""
        if self.model is None:
            return None
        if isinstance(self.model, dict):
            return self.model                                                             # raw JSON schema
        # Pydantic model → schema
        return self.model.model_json_schema()

    @classmethod
    def parse_json(cls: type[T], json_data: str | bytes | bytearray) -> T:
        return cls.model_validate_json(json_data)

    @classmethod
    def from_ollama_response(cls: type[T], response: "OllamaResponse | str") -> T:
        text = response.text if isinstance(response, OllamaResponse) else response
        return cls.parse_json(text)

    # ----------------------------------------------------------------------
    # Message & Input (core building blocks)
    # ----------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaMessage:
    """Ollama message (identical public API to xAIMessage)."""

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
    # Request
    # ----------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaRequest(BaseModel):
    """Ollama-native request (mirrors xAIRequest API)."""

    model: str
    input: OllamaInput | str = Field(
        ..., description="Accepts str or OllamaInput for legacy compatibility"
    )
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None                                                         # maps to 'num_predict' in Ollama options
    response_format: OllamaJSONResponseSpec | None = None
    tools: list[dict[str, Any]] | None = None
    keep_alive: str | int | None = None                                                   # e.g. "5m"
    save_mode: SaveMode = "none"
    prompt_cache_key: str | None = None                                                   # not used by Ollama but kept for API parity

    model_config = ConfigDict(frozen=True)

    def __post_init__(self) -> None:
        if self.response_format is not None:
            self._validate_json_instruction_present()

    def _validate_json_instruction_present(self) -> None:
        # Ollama does NOT require a magic string like xAI, but we keep the method for parity
        pass

    def get_messages(self) -> list[dict[str, Any]]:
        """Always returns normalised list of dicts (same contract as xAIRequest)."""
        if isinstance(self.input, str):
            return [{"role": "user", "content": self.input}]
        return self.input.to_list()

    def to_ollama_dict(self) -> dict[str, Any]:
        """Exact payload for Ollama /api/chat or /api/generate."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.get_messages(),
            "stream": False,                                                              # overridden by caller when streaming
        }

        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
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

    def has_media(self) -> bool:
        """True if any user message contains base64 images."""
        for msg in self.get_messages():
            if msg.get("role") == "user" and msg.get("images"):
                return True
        return False

    def extract_prompt_snippet(self, max_chars: int = 100) -> str:
        """Same helper as xAI side (used by persistence)."""
        msgs = self.get_messages()
        for m in msgs:
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    return content[:max_chars]
                # multimodal text parts
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return " ".join(text_parts)[:max_chars]
        return ""

    def with_updates(self, **updates: Any) -> "OllamaRequest":
        return replace(self, **updates)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OllamaRequest":
        data = dict(data)
        if "messages" in data and "input" not in data:
            data["input"] = OllamaInput.from_list(data.pop("messages"))
        elif isinstance(data.get("input"), (list, tuple)):
            data["input"] = OllamaInput.from_list(data["input"])
        return cls(**data)

    # ----------------------------------------------------------------------
    # Response & Streaming Chunk
    # ----------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaResponse(BaseModel):
    """Ollama-native response (non-streaming or final streaming chunk)."""

    model: str
    created_at: str
    message: dict[str, Any]                                                               # contains role, content, tool_calls, images
    done: bool = True
    done_reason: DoneReason | None = None

    # Ollama local performance telemetry (huge value for local models)
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None

    raw: dict[str, Any] = field(default_factory=dict)
    parsed: BaseModel | dict | None = None                                                # for structured output

    @property
    def text(self) -> str:
        """Main assistant content (mirrors xAIResponse.text)."""
        return self.message.get("content", "") or ""

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return self.message.get("tool_calls") or []

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


@dataclass(frozen=True)
class OllamaStreamingChunk:
    """Implements LLMStreamingChunkProtocol — drop-in compatible with xAI streaming code."""

    text: str = ""
    finish_reason: str | None = None
    tool_calls_delta: list[dict[str, Any]] | None = None
    is_final: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    # Ollama-specific convenience fields (populated on final chunk)
    done_reason: DoneReason | None = None
    total_duration: int | None = None

    # Convenience factory


def parse_ollama_response(data: dict[str, Any]) -> OllamaResponse:
    """Helper used by the future Ollama client layer."""
    return OllamaResponse.from_dict(data)


# ----------------------------------------------------------------------
# Embeddings Support (Ollama /api/embed)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaEmbedRequest:
    """Lightweight request model for Ollama embeddings.

    Fully supports the official /api/embed contract.

    Attributes
    ----------
    model : str
        Name of the embedding model (e.g. "nomic-embed-text", "mxbai-embed-large").
    input : str | Sequence[str]
        Single text or list of texts to embed.
        Equivalent to NumPy: shape (n_inputs,) of strings.
    truncate : bool, default True
        Truncate inputs that exceed the model's context length.
    options : dict[str, Any] | None, default None
        Model-specific options (rarely needed for embeddings).
    keep_alive : str | int | None, default None
        How long to keep the model loaded (e.g. "5m", 300).
    dimensions : int | None, default None
        Target embedding dimension (supported by some models for reduction).
    """

    model: str
    input: str | Sequence[str]
    truncate: bool = True
    options: dict[str, Any] | None = None
    keep_alive: str | int | None = None
    dimensions: int | None = None

    def to_ollama_dict(self) -> dict[str, Any]:
        """Build the exact JSON payload for POST /api/embed."""
        payload: dict[str, Any] = {
            "model": self.model,
            "input": self.input if isinstance(self.input, str) else list(self.input),
            "truncate": self.truncate,
        }
        if self.options:
            payload["options"] = self.options
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        return {k: v for k, v in payload.items() if v is not None}


@dataclass(frozen=True)
class OllamaEmbedResponse:
    """Lightweight response model for Ollama embeddings.

    Attributes
    ----------
    model : str
        Embedding model used.
    embeddings : list[list[float]]
        The generated vectors.
        **NumPy notation**: equivalent to `np.ndarray` of shape `(n_inputs, embedding_dim)`.
        Each inner list is one embedding vector (float32/64 depending on model).
    total_duration : int | None
        Total time in nanoseconds (includes model load + inference).
    load_duration : int | None
        Time to load the model into memory (ns).
    prompt_eval_count : int | None
        Number of tokens evaluated in the prompt(s).
    prompt_eval_duration : int | None
        Time spent evaluating the prompt(s) (ns).
    raw : dict[str, Any]
        Full raw response from Ollama (for debugging/advanced use).
    """

    model: str
    embeddings: list[list[float]]
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def n_inputs(self) -> int:
        """Number of input texts embedded (shape[0] in NumPy terms)."""
        return len(self.embeddings)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of each vector (shape[1] in NumPy terms)."""
        return len(self.embeddings[0]) if self.embeddings else 0

    def to_numpy(self) -> "np.ndarray":
        """Convert to NumPy array (lazy import — numpy is optional).

        Returns
        -------
        np.ndarray
            Shape: (n_inputs, embedding_dim), dtype=float32 or float64.
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "numpy is required for to_numpy(). " "Install with: pip install numpy"
            ) from e
        return np.array(self.embeddings, dtype=np.float32)

    def cosine_similarity(self, idx1: int, idx2: int) -> float:
        """Compute cosine similarity between two embeddings (SciPy notation).

        Uses scipy.spatial.distance.cosine under the hood (lazy import).

        Parameters
        ----------
        idx1, idx2 : int
            Indices into self.embeddings.

        Returns
        -------
        float
            Cosine similarity in [-1, 1].
        """
        try:
            from scipy.spatial.distance import cosine
        except ImportError as e:
            raise ImportError(
                "scipy is required for cosine_similarity(). "
                "Install with: pip install scipy"
            ) from e
        vec1 = self.embeddings[idx1]
        vec2 = self.embeddings[idx2]
        return 1.0 - cosine(vec1, vec2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OllamaEmbedResponse":
        return cls(
            model=data["model"],
            embeddings=data.get("embeddings", []),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            prompt_eval_duration=data.get("prompt_eval_duration"),
            raw=data,
        )
