"""
Generic LLM Request, Response, and Streaming Chunk Protocols + SaveMode.

These protocols define a minimal, provider-agnostic interface that every
request, response, and streaming chunk object must implement.

Design goals:
- Zero knowledge of xAI vs Ollama (or future providers) in persistence, logging, etc.
- Consistent API across all providers
- Easy to add new providers (just implement the three methods/properties)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


# ----------------------------------------------------------------------
# SaveMode - now centralised here (moved from xai_objects.py)
# ----------------------------------------------------------------------
type SaveMode = Literal["none", "json_files", "postgres"]


# ----------------------------------------------------------------------
# LLMRequestProtocol
# ----------------------------------------------------------------------
@runtime_checkable
class LLMRequestProtocol(Protocol):
    """Minimal contract every request object must satisfy."""

    def meta(self) -> dict[str, Any]:
        """Return generation settings (temperature, max_tokens, save_mode, etc.)."""
        ...

    def payload(self) -> dict[str, Any]:
        """Return the actual prompt / messages / input."""
        ...

    def endpoint(self) -> "LLMEndpoint":
        """Return provider + model + connection information."""
        ...


# ----------------------------------------------------------------------
# LLMResponseProtocol
# ----------------------------------------------------------------------
@runtime_checkable
class LLMResponseProtocol(Protocol):
    """Minimal contract every response object must satisfy."""

    def meta(self) -> dict[str, Any]:
        """Return generation settings that were actually used."""
        ...

    def payload(self) -> dict[str, Any]:
        """Return the generated output (text, tool_calls, parsed, etc.)."""
        ...

    def endpoint(self) -> "LLMEndpoint":
        """Return provider + model + connection information."""
        ...


# ----------------------------------------------------------------------
# LLMStreamingChunkProtocol - now centralised here (moved from xai_objects.py)
# ----------------------------------------------------------------------
@runtime_checkable
class LLMStreamingChunkProtocol(Protocol):
    """Common contract for streaming chunks across all providers.

    Uses properties for simplicity (streaming chunks are lightweight).
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


# ----------------------------------------------------------------------
# LLMEndpoint
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LLMEndpoint:
    """Structured, validated representation of an LLM endpoint."""

    provider: Literal["ollama", "xai", "groq", "anthropic", "openai", "vllm", "other"]
    model: str
    base_url: str | None = None
    path: str | None = None
    api_type: Literal["native", "sdk", "openai_compat", "batch"] | None = None
    extra: Any = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"provider": self.provider, "model": self.model}
        if self.base_url:
            d["base_url"] = self.base_url
        if self.path:
            d["path"] = self.path
        if self.api_type:
            d["api_type"] = self.api_type
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMEndpoint":
        return cls(
            provider=data["provider"],
            model=data["model"],
            base_url=data.get("base_url"),
            path=data.get("path"),
            api_type=data.get("api_type"),
            extra=data.get("extra", {}),
        )
