"""
Generic LLM Request and Response Protocols for the ai_api package.

These protocols define a minimal, provider-agnostic interface that every
request and response object (xAI, Ollama, embeddings, future providers)
must implement. This allows persistence.py, logging, and other cross-cutting
concerns to work with plain dictionaries and never need to know the concrete
class types.

Design goals:
- Zero knowledge of xAI vs Ollama in persistence layer
- Consistent `meta()`, `payload()`, `endpoint()` API across all providers
- All methods return vanilla Python dicts for maximum decoupling
- Easy to add new providers (just implement the three methods)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@runtime_checkable
class LLMRequestProtocol(Protocol):
    """Minimal contract every request object must satisfy.

    Implementations:
        - xAIRequest
        - OllamaRequest
        - OllamaEmbedRequest (and future embedding requests)
        - Any new provider request
    """

    def meta(self) -> dict[str, Any]:
        """Return generation settings that drove the response.

        Typical keys: temperature, max_tokens, response_format, tools,
        keep_alive, truncate, dimensions, save_mode, etc.
        """
        ...

    def payload(self) -> dict[str, Any]:
        """Return the actual prompt / messages / input being sent to the model.

        For chat:   {"messages": [...], "input_type": "chat"}
        For raw:    {"prompt": "...", "input_type": "raw"}
        For embed:  {"input": ["text1", "text2"], "input_type": "embeddings"}
        """
        ...

    def endpoint(self) -> "LLMEndpoint":
        """Return provider + model + connection information.

        Example:  TODO - UPDATE TO SHOW LLMEndpoint.
            {
                "provider": "ollama" | "xai",
                "model": "llama3.2" | "grok-4",
                "base_url": "http://localhost:11434" | "https://api.x.ai/v1",
                "path": "/api/chat" | "/chat/completions",
                "api_type": "native" | "sdk"
            }
        """
        ...


@runtime_checkable
class LLMResponseProtocol(Protocol):
    """Minimal contract every response object must satisfy.

    Implementations:
        - xAIResponse
        - OllamaResponse
        - OllamaEmbedResponse (and future embedding responses)
        - Any new provider response
    """

    def meta(self) -> dict[str, Any]:
        """Return generation settings that were actually used (echoed back)."""
        ...

    def payload(self) -> dict[str, Any]:
        """Return the generated output.

        For chat:   {"text": "...", "tool_calls": [...], "finish_reason": "..."}
        For embed:  {"embeddings": [[...], [...]], "n_inputs": 2, "embedding_dim": 768}
        """
        ...

    def endpoint(self) -> "LLMEndpoint":
        """Return provider + model + connection information (same shape as request).

        Recommended: return an LLMEndpoint instance for richer typing and validation.
        """
        ...

        # ----------------------------------------------------------------------
        # LLMEndpoint - rich, typed endpoint descriptor
        # ----------------------------------------------------------------------


@dataclass(frozen=True)
class LLMEndpoint:
    """Structured, validated representation of an LLM endpoint.

    Use this in `endpoint()` methods for better type safety and richer metadata
    than a plain dict. All fields are optional except provider + model.

    Example usage in a request class:
        def endpoint(self) -> LLMEndpoint:
            return LLMEndpoint(
                provider="ollama",
                model=self.model,
                base_url="http://localhost:11434",
                path="/api/chat",
                api_type="native",
                extra={"timeout": 180}
            )
    """

    provider: Literal["ollama", "xai", "groq", "anthropic", "openai", "vllm", "other"]
    model: str
    base_url: str | None = None
    path: str | None = None
    api_type: Literal["native", "sdk", "openai_compat", "batch"] | None = None
    extra: Any = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict (useful for persistence / logging)."""
        d = {
            "provider": self.provider,
            "model": self.model,
        }
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
