#!/usr/bin/env python3
"""
Unified response abstraction for all LLM providers.

This module provides `LLMResponse` (immutable result from any provider)
and `StreamingChunk` (for the streaming interface requested). The
`continuation_token` abstraction unifies Grok's `x-grok-conv-id` with
Ollama's native `context` list, enabling seamless multi-turn caching
across local and remote models.

All persistence, logging, and dashboard extraction layers will use only
these unified objects — no provider-specific code leaks into them.

Streaming support is included here (as a lightweight dataclass) so that
`submit_batch` and `stream` share the same response vocabulary.
"""

from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..data_structures.LLM_types_grok import (
    GrokResponse,                                                                         # reuse existing parser logic where possible
)

type ProviderLiteral = Literal["grok", "ollama"]
type ContinuationToken = str | list[int] | bytes | None


@dataclass(frozen=True)
class LLMResponse:
    """
    Immutable response from any LLM provider.

    Parameters
    ----------
    id : str
        Unique response identifier returned by the provider.
    created_at : int
        Unix timestamp of response creation.
    model : str
        Model that produced the response.
    provider : ProviderLiteral
        "grok" or "ollama" (used for routing in persistence/dashboard).
    text : str
        Concatenated final text output (convenience property).
    tool_calls : list[dict[str, Any]]
        Any tool calls present in the response.
    usage : dict[str, Any] | None
        Token counts (prompt_tokens, completion_tokens, etc.).
    continuation_token : ContinuationToken
        Opaque token for cheap multi-turn continuation:
        - Grok  → str (x-grok-conv-id)
        - Ollama → list[int] (native context array)
    raw : dict[str, Any]
        Original provider payload (for debugging / forward compatibility).
    extra : dict[str, Any]
        Provider-specific fields (Ollama timings, Grok reasoning, …).

    Notes
    -----
    - Frozen for safety in async batches and logging.
    - `.text` and `.tool_calls` properties mirror existing GrokResponse.
    """

    id: str
    created_at: int
    model: str
    provider: ProviderLiteral
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    continuation_token: ContinuationToken = None
    raw: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls,
        raw: dict[str, Any],
        provider: ProviderLiteral,
        continuation_token: ContinuationToken = None,
    ) -> "LLMResponse":
        """
        Factory that converts a raw provider response into the unified shape.

        Parameters
        ----------
        raw : dict[str, Any]
            JSON returned by the concrete client (Grok or Ollama).
        provider : ProviderLiteral
            Which back-end produced the response.
        continuation_token : ContinuationToken, optional
            Pre-extracted token (Grok header or Ollama context).

        Returns
        -------
        LLMResponse
            Fully populated immutable instance.

        Flow
        ----
        1. Extract common fields.
        2. Use GrokResponse parser for Grok (reuse existing logic).
        3. For Ollama, map native fields directly.
        4. Store everything in .raw and .extra for full fidelity.
        """
        if provider == "grok":
            grok_resp = GrokResponse.from_dict(raw)                                       # reuse existing parser
            return cls(
                id=grok_resp.id,
                created_at=grok_resp.created_at,
                model=grok_resp.model,
                provider=provider,
                text=grok_resp.text,
                tool_calls=grok_resp.tool_calls,
                usage=grok_resp.usage,
                continuation_token=continuation_token,
                raw=raw,
                extra={},
            )

        # Ollama path (native or OpenAI-compat)
        output = raw.get("message", {}).get("content", "") or raw.get("response", "")
        return cls(
            id=raw.get("id", "ollama-unknown"),
            created_at=raw.get("created_at", 0),
            model=raw.get("model", "unknown"),
            provider=provider,
            text=output,
            tool_calls=raw.get("message", {}).get("tool_calls", []),
            usage=raw.get("usage"),
            continuation_token=raw.get("context") or continuation_token,
            raw=raw,
            extra={
                "total_duration": raw.get("total_duration"),
                "load_duration": raw.get("load_duration"),
                "eval_count": raw.get("eval_count"),
            },
        )


@dataclass(frozen=True)
class StreamingChunk:
    """
    Single token (or partial tool call) yielded during streaming.

    Used by `client.stream(...)` for both Grok and Ollama. Keeps the
    streaming interface identical regardless of provider.
    """

    delta_text: str
    finished: bool = False
    tool_call_delta: dict[str, Any] | None = None
    usage_partial: dict[str, Any] | None = None
    raw_chunk: Any = None
