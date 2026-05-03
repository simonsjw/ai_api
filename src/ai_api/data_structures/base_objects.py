"""
Generic LLM Request, Response, and Streaming Chunk Protocols + SaveMode + Neutral Chat Formats.

This module provides the foundational, provider-agnostic abstractions for the
ai_api package. It defines minimal Protocol interfaces (using structural
subtyping / duck typing) that every concrete request, response, and streaming
chunk implementation must satisfy, regardless of whether the backend is
Ollama (local), xAI (remote), or future providers (Groq, Anthropic, etc.).

It also introduces a set of Pydantic-based neutral data models used for
branch-aware, zero-duplication chat persistence in Postgres. These models
(NeutralTurn, NeutralPrompt, NeutralResponseBlob) allow every provider to
store and reconstruct conversation history in a single, consistent shape
while the actual branching tree is maintained via the database columns
``tree_id``, ``branch_id``, ``parent_response_id``, and ``sequence``.

**What it does:**
- Centralises the "contract" so higher-level code (chat_turn_*.py, persistence,
  logging, structured output parsers) can work uniformly without knowing the
  provider.
- Defines ``LLMEndpoint`` as a frozen, validated container for connection
  details (provider name, model, base_url, api_type, etc.).
- Introduces ``SaveMode`` type alias used across the package for persistence
  configuration ("none", "json_files", "postgres").
- Provides neutral-format models so that ``persist_chat_turn`` can store
  only the *delta* (last prompt + generated response) while full history is
  reconstructed on demand via recursive CTE on the parent links.

**How it does it:**
- Uses ``@runtime_checkable`` Protocols so ``isinstance(obj, LLMRequestProtocol)``
  works at runtime even for classes that never inherit from it (pure structural
  typing).
- All methods are deliberately lightweight (return dicts or simple objects)
  to avoid heavy dependencies in the base layer.
- ``LLMEndpoint`` is a ``dataclass(frozen=True)`` with ``to_dict`` / ``from_dict``
  for easy (de)serialisation into the Postgres JSONB columns and for logging.
- Neutral models are Pydantic ``BaseModel`` subclasses so they get automatic
  validation, serialisation, and JSON schema generation for the ``response``
  JSONB column.

The design guarantees zero knowledge of xAI vs Ollama specifics in the
persistence, streaming, or batch layers — new providers only need to implement
the three protocol methods/properties plus the two conversion methods
(``from_neutral_history`` and ``to_neutral_format``) to participate in the
branched-chat system.

Examples
--------
Creating objects that can be sent to an LLM (via higher-level client code):

>>> from src.ai_api.data_structures.base_objects import LLMEndpoint, SaveMode
>>> endpoint = LLMEndpoint(
...     provider="ollama",
...     model="llama3.2",
...     base_url="http://localhost:11434",
...     path="/api/chat",
...     api_type="native",
... )
>>> print(endpoint.to_dict())
{'provider': 'ollama', 'model': 'llama3.2', 'base_url': 'http://localhost:11434', ...}

Processing responses (media files & embeddings are handled in concrete
implementations that satisfy the protocol; the base only guarantees the
meta/payload/endpoint shape):

>>> # After an LLM call returns an object satisfying LLMResponseProtocol
>>> response: LLMResponseProtocol = ...  # e.g. OllamaResponse or xAIResponse
>>> meta = response.meta()  # contains usage, telemetry, finish_reason
>>> payload = response.payload()  # contains 'text', 'tool_calls', 'parsed', media refs
>>> if "images" in payload.get("raw", {}):  # media handling example
...     print("Response contained generated images or references")
>>> # Embeddings would appear in payload['parsed'] or tool_calls if the
>>> # model was instructed to return them (e.g. via JSON schema)

For Ollama-specific monitoring of software/hardware (see ollama_objects.py
for full telemetry):
The base protocol ensures that ``meta()`` and ``payload()`` always expose
the rich local stats that Ollama returns (total_duration, eval_count, etc.)
so monitoring dashboards can be written once against the protocol.

Branched-chat persistence example (new in this version):

>>> from src.ai_api.core.common.persistence import PersistenceManager
>>> pm = PersistenceManager(logger=logger, db_url=DB_URL)
>>> # Reconstruct only the relevant slice of history for a fork
>>> history = await pm.reconstruct_neutral_branch(
...     tree_id=some_tree_id,
...     branch_id=some_branch_id,
...     start_from_response_id=parent_id,
...     max_depth=50,
... )
>>> # Convert neutral history + new prompt into a provider-specific request
>>> req = OllamaRequest.from_neutral_history(history, "Explain the algorithm", meta)
>>> resp = await client._call_ollama(req)
>>> # Persist only the delta; full tree is reconstructed via parent_response_id
>>> saved = await pm.persist_chat_turn(resp, req, tree_id=some_tree_id, ...)

See Also
--------
ollama_objects.OllamaRequest, xai_objects.xAIRequest : concrete implementations
core.common.persistence.PersistenceManager : branch-aware persistence layer
core.common.chat_session.ChatSession : high-level orchestrator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# EXISTING CLASSES (unchanged — @dataclass(frozen=True) auto-generates __init__)
# ─────────────────────────────────────────────────────────────────────────────


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


@runtime_checkable
class LLMRequestProtocol(Protocol):
    def endpoint(self) -> LLMEndpoint: ...
    def meta(self) -> dict[str, Any]: ...
    def payload(self) -> dict[str, Any]: ...


class LLMResponseProtocol(Protocol):
    def endpoint(self) -> LLMEndpoint: ...
    def meta(self) -> dict[str, Any]: ...
    def payload(self) -> dict[str, Any]: ...
    def to_neutral_format(self, branch_info: dict | None = None) -> dict: ...


class LLMStreamingChunkProtocol(Protocol):
    text: str
    finish_reason: str | None
    tool_calls_delta: list[dict] | None
    is_final: bool
    raw: dict[str, Any]


SaveMode = Literal["none", "json_files", "postgres"]


def utc_now() -> datetime:
    """Return current UTC time as an aware datetime."""
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: NEUTRAL FORMATS FOR BRANCH-AWARE PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────


class NeutralTurn(BaseModel):
    """
    A single turn in the neutral, provider-agnostic chat format.

    This model is the canonical representation used inside the ``response``
    JSONB column of the ``responses`` table and when reconstructing history
    via ``reconstruct_neutral_branch``. Every provider-specific response is
    converted to this shape before being written to Postgres, guaranteeing
    that the database never contains provider-specific field names.

    Parameters
    ----------
    role : {"system", "user", "assistant", "tool", "developer"}
        Speaker of the turn. Follows the OpenAI / Ollama / xAI convention.
    content : str or list of dict, optional
        The textual (or multimodal) content of the message. For assistant
        turns this is the generated text; for user turns it is the prompt.
        Multimodal content is stored as a list of content blocks
        (e.g. ``[{"type": "text", "text": "..."}, {"type": "image_url", ...}]``).
    images : list of str, optional
        Base64-encoded images or URLs attached to the turn (primarily for
        vision models). Only populated on user or assistant turns that
        contain visual input/output.
    structured : dict, optional
        Parsed structured output when the request used ``response_format``
        (Pydantic model or JSON schema). Stored exactly as returned by the
        provider so that downstream consumers can re-instantiate the original
        Pydantic instance if desired.
    tools : list of dict, optional
        Tool / function-call specifications or results. For assistant turns
        this contains the calls the model wants to make; for tool turns it
        contains the execution results.
    finish_reason : str, optional
        Why generation stopped ("stop", "length", "tool_calls", "content_filter",
        etc.). Only meaningful on the final assistant turn of a request.
    usage : dict, optional
        Token usage and timing statistics returned by the provider
        (prompt_tokens, completion_tokens, total_duration, etc.).
    raw : dict, optional
        The original, unmodified payload returned by the provider. Useful
        for debugging, telemetry, and future-proofing when new fields appear.
    timestamp : datetime, optional
        UTC timestamp when the turn was created. Defaults to ``datetime.utcnow()``.
    branch_meta : dict, optional
        Branch-specific metadata injected by ``persist_chat_turn``:
        ``{"tree_id": "...", "branch_id": "...", "parent_response_id": "...",
        "sequence": 7}``. This allows a consumer to know exactly where in the
        conversation tree the turn belongs without having to query the
        relational columns.

    See Also
    --------
    NeutralPrompt : the prompt that produced this turn.
    NeutralResponseBlob : the wrapper stored in the ``response`` JSONB column.
    """

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]]
    images: list[str] | None = None
    structured: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    timestamp: datetime | None = Field(default_factory=utc_now)
    branch_meta: dict[str, Any] | None = None


class NeutralPrompt(BaseModel):
    """
    The exact prompt (system + user) that produced a particular assistant turn.

    Stored inside ``NeutralResponseBlob.prompt`` so that the database row
    for a response contains both the input that generated it and the output,
    without duplicating the entire conversation history.

    Parameters
    ----------
    system : str, optional
        The system prompt / developer instruction that was active for this turn.
        May be ``None`` when the conversation has no system message.
    user : str or list of dict
        The user message that triggered the assistant response. Can be plain
        text or a multimodal content list (same shape as ``NeutralTurn.content``).
    structured_spec : dict, optional
        The ``response_format`` specification that was sent to the model
        (either a Pydantic model class or a raw JSON schema dict). Used by
        the structured-output parser after the call returns.
    tools : list of dict, optional
        Tool / function definitions that were available to the model for
        this turn (the ``tools`` parameter of the chat completion request).

    See Also
    --------
    NeutralTurn : the assistant response that this prompt produced.
    NeutralResponseBlob : the container written to the ``response`` JSONB column.
    """

    system: str | None = None
    user: str | list[dict[str, Any]]
    structured_spec: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None


class NeutralResponseBlob(BaseModel):
    """
    The complete payload written to the ``response`` JSONB column.

    Each row in the ``responses`` table stores exactly one turn. The
    ``response`` column contains this blob, which in turn holds the prompt
    that produced the turn and the resulting assistant message. All other
    settings (temperature, usage, model, etc.) live in the sibling ``meta``
    JSONB column. This design keeps the branching metadata (tree_id,
    parent_response_id, sequence) in relational columns while the actual
    conversational content stays in a single, queryable JSONB field.

    Parameters
    ----------
    prompt : NeutralPrompt
        The system and user messages that were sent to the model for this turn.
    response : NeutralTurn
        The assistant (or tool) response that was generated, already converted
        to neutral format via the provider's ``to_neutral_format`` method.
    branch_context : dict
        Copy of the branching identifiers that were also written to the
        relational columns. Redundant but extremely convenient for ad-hoc
        JSON queries and for clients that only have access to the JSONB blob.

    See Also
    --------
    NeutralTurn, NeutralPrompt : the two components that make up the blob.
    PersistenceManager.persist_chat_turn : the method that constructs and
        inserts this blob together with the relational branching columns.
    """

    prompt: NeutralPrompt
    response: NeutralTurn
    branch_context: dict[str, Any] = Field(default_factory=dict)


# TypedDict for lighter internal use if preferred
class NeutralTurnDict(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None
    images: list[str] | None
    structured: dict[str, Any] | None
    tools: list[dict[str, Any]] | None
    finish_reason: str | None
    usage: dict[str, Any] | None
    raw: dict[str, Any] | None
    timestamp: str | None
    branch_meta: dict[str, Any] | None


# Re-export for convenience
__all__ = [
    "LLMEndpoint",
    "LLMRequestProtocol",
    "LLMResponseProtocol",
    "LLMStreamingChunkProtocol",
    "SaveMode",
    "NeutralTurn",
    "NeutralPrompt",
    "NeutralResponseBlob",
    "NeutralTurnDict",
]
