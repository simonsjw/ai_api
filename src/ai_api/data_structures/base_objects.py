"""
Generic LLM Request, Response, and Streaming Chunk Protocols + SaveMode.

This module provides the foundational, provider-agnostic abstractions for the
ai_api package. It defines minimal Protocol interfaces (using structural
subtyping / duck typing) that every concrete request, response, and streaming
chunk implementation must satisfy, regardless of whether the backend is
Ollama (local), xAI (remote), or future providers (Groq, Anthropic, etc.).

**What it does:**
- Centralises the "contract" so higher-level code (chat_turn_*.py, persistence,
  logging, structured output parsers) can work uniformly without knowing the
  provider.
- Defines ``LLMEndpoint`` as a frozen, validated container for connection
  details (provider name, model, base_url, api_type, etc.).
- Introduces ``SaveMode`` type alias used across the package for persistence
  configuration ("none", "json_files", "postgres").

**How it does it:**
- Uses ``@runtime_checkable`` Protocols so ``isinstance(obj, LLMRequestProtocol)``
  works at runtime even for classes that never inherit from it (pure structural
  typing).
- All methods are deliberately lightweight (return dicts or simple objects)
  to avoid heavy dependencies in the base layer.
- ``LLMEndpoint`` is a ``dataclass(frozen=True)`` with ``to_dict`` / ``from_dict``
  for easy (de)serialisation into the Postgres JSONB columns and for logging.

The design guarantees zero knowledge of xAI vs Ollama specifics in the
persistence, streaming, or batch layers — new providers only need to implement
the three protocol methods/properties.

Examples
--------
Creating objects that can be sent to an LLM (via higher-level client code):

>>> from src.ai_api.data_structures.base_objects import LLMEndpoint, SaveMode
>>> endpoint = LLMEndpoint(
...     provider="ollama",
...     model="llama3.2",
...     base_url="http://localhost:11434",
...     path="/api/chat",
...     api_type="native"
... )
>>> print(endpoint.to_dict())
{'provider': 'ollama', 'model': 'llama3.2', 'base_url': 'http://localhost:11434', ...}

Processing responses (media files & embeddings are handled in concrete
implementations that satisfy the protocol; the base only guarantees the
meta/payload/endpoint shape):

>>> # After an LLM call returns an object satisfying LLMResponseProtocol
>>> response: LLMResponseProtocol = ...  # e.g. OllamaResponse or xAIResponse
>>> meta = response.meta()          # contains usage, telemetry, finish_reason
>>> payload = response.payload()    # contains 'text', 'tool_calls', 'parsed', media refs
>>> if 'images' in payload.get('raw', {}):  # media handling example
...     print("Response contained generated images or references")
>>> # Embeddings would appear in payload['parsed'] or tool_calls if the
>>> # model was instructed to return them (e.g. via JSON schema)

For Ollama-specific monitoring of software/hardware (see ollama_objects.py
for full telemetry):
The base protocol ensures that ``meta()`` and ``payload()`` always expose
the rich local stats that Ollama returns (total_duration, eval_count, etc.)
so monitoring dashboards can be written once against the protocol.

See Also
--------
ollama_objects.OllamaRequest, xai_objects.xAIRequest : concrete implementations
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
    """Minimal contract every request object must satisfy.

    Any class that implements ``meta()``, ``payload()``, and ``endpoint()``
    is automatically treated as a valid request (no inheritance required).

    This enables the rest of the package (persistence, streaming clients,
    batch processors) to remain completely provider-agnostic.

    Methods
    -------
    meta : () -> dict[str, Any]
        Return generation settings (temperature, max_tokens, save_mode, etc.).
        Used for logging, DB storage, and cache-key generation.
    payload : () -> dict[str, Any]
        Return the actual prompt / messages / input that will be sent to the
        model. For chat models this is usually {"messages": [...] }.
    endpoint : () -> LLMEndpoint
        Return provider + model + connection information. Used to route the
        request and to populate the ``endpoint`` JSONB column.

    Examples
    --------
    >>> req: LLMRequestProtocol = OllamaRequest(model="llama3", input="Hello")
    >>> print(req.meta()["temperature"])
    0.7
    >>> print(req.endpoint().provider)
    'ollama'
    """

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
    """Minimal contract every response object must satisfy.

    Guarantees that every provider returns at least ``meta()``, ``payload()``
    and ``endpoint()`` in a consistent shape. This is what the DB writer,
    structured-output parser, and streaming aggregator all consume.

    Methods
    -------
    meta : () -> dict[str, Any]
        Return generation settings that were actually used (may differ from
        request if the server overrode anything) plus usage statistics.
    payload : () -> dict[str, Any]
        Return the generated output (text, tool_calls, parsed, media refs,
        embeddings if requested, etc.).
    endpoint : () -> LLMEndpoint
        Return provider + model + connection information (same as request).

    Examples
    --------
    Processing a response that may contain media or embeddings:

    >>> resp: LLMResponseProtocol = get_response_from_llm(...)
    >>> payload = resp.payload()
    >>> print(payload["text"][:100])
    'The image shows a cat...'
    >>> if payload.get("parsed"):
    ...     # embeddings or structured data returned by model
    ...     emb = payload["parsed"].get("embedding")
    ...     print("Embedding dimension:", len(emb) if emb else 0)
    >>> # Media files are referenced in raw or via tool_calls / attachments
    """

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

    Uses properties for simplicity (streaming chunks are lightweight and
    frequently created). All concrete chunks (OllamaStreamingChunk,
    xAIStreamingChunk) implement these exact attributes.

    Attributes
    ----------
    text : str
        Incremental text delta (can be empty on tool-call or final chunks).
    finish_reason : str | None
        "stop", "length", "tool_calls", etc. Only present on final chunk.
    tool_calls_delta : list[dict] | None
        Partial tool-call data (for function-calling streaming).
    is_final : bool
        True only for the last chunk of a stream.
    raw : dict[str, Any]
        The original provider payload (useful for debugging/telemetry).

    Examples
    --------
    >>> chunk: LLMStreamingChunkProtocol = stream.__anext__()
    >>> if chunk.is_final:
    ...     print("Stream finished with reason:", chunk.finish_reason)
    >>> print(chunk.text, end="")
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
    """Structured, validated representation of an LLM endpoint.

    Used everywhere a request or response needs to record "which model on
    which provider". Stored as JSONB in Postgres and used for routing,
    logging, and cache invalidation.

    Parameters
    ----------
    provider : {"ollama", "xai", "groq", "anthropic", "openai", "vllm", "other"}
        Logical provider name. "ollama" implies local HTTP, "xai" implies
        the official SDK / https://api.x.ai.
    model : str
        Model identifier as accepted by the provider (e.g. "llama3.2",
        "grok-2-latest").
    base_url : str or None, optional
        Override the default base URL. For Ollama usually
        "http://localhost:11434"; for xAI the SDK ignores this.
    path : str or None, optional
        API path ("/api/chat" for Ollama native, "/chat/completions" for
        OpenAI-compatible).
    api_type : {"native", "sdk", "openai_compat", "batch"} or None, optional
        How the client should call the endpoint.
    extra : dict, optional
        Provider-specific extra configuration (e.g. headers, timeout).

    Methods
    -------
    to_dict
        Convert to a plain dict suitable for JSONB storage or logging.
    from_dict
        Reconstruct from a dict (used when reading from DB or cache).

    Examples
    --------
    >>> ep = LLMEndpoint(provider="ollama", model="llama3.2")
    >>> ep.to_dict()
    {'provider': 'ollama', 'model': 'llama3.2'}
    >>> LLMEndpoint.from_dict({"provider": "xai", "model": "grok-2"})
    LLMEndpoint(provider='xai', model='grok-2', ...)
    """

    provider: Literal["ollama", "xai", "groq", "anthropic", "openai", "vllm", "other"]
    model: str
    base_url: str | None = None
    path: str | None = None
    api_type: Literal["native", "sdk", "openai_compat", "batch"] | None = None
    extra: Any = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert endpoint to a JSON-serialisable dict.

        Returns
        -------
        dict[str, Any]
            Minimal dict with only non-None fields.
        """
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
        """Reconstruct an LLMEndpoint from a dict (e.g. from Postgres JSONB).

        Parameters
        ----------
        data : dict[str, Any]
            Must contain at least "provider" and "model".

        Returns
        -------
        LLMEndpoint
            Frozen instance.
        """
        return cls(
            provider=data["provider"],
            model=data["model"],
            base_url=data.get("base_url"),
            path=data.get("path"),
            api_type=data.get("api_type"),
            extra=data.get("extra", {}),
        )
