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

import re
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
    reasoning_content: Extracted chain-of-thought / reasoning trace when
        capture_reasoning=True.

    Notes
    -----
    - Frozen for safety in async batches and logging.
    - `.text` and `.tool_calls` properties mirror existing GrokResponse.
    - For Grok-4 reasoning_content may be encrypted (logged as warning).
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
    reasoning_content: str | None = None

    @classmethod
    def from_raw(
        cls,
        raw: dict[str, Any],
        provider: ProviderLiteral,
        continuation_token: ContinuationToken = None,
        capture_reasoning: bool = False,
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
        reasoning_content: bool, default False
            Return the reasoning content if available.

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
        reasoning_content: str | None = None
        output: str = ""

        if capture_reasoning:
            if provider == "grok":
                msg = raw.get("choices", [{}])[0].get("message", {})
                reasoning_content = msg.get("reasoning_content") or msg.get(
                    "reasoning", {}
                ).get("encrypted_content")
                if reasoning_content and "encrypted" in str(reasoning_content).lower():
                    print(
                        "Grok-4 returned encrypted reasoning_content"
                    )                                                                     # was logger.warning.
            else:                                                                         # Ollama
                # Get full content (works for both native and OpenAI-compat)
                content = raw.get("choices", [{}])[0].get("message", {}).get(
                    "content", ""
                ) or raw.get("response", "")
                match = re.search(
                    r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE
                )
                if match:
                    reasoning_content = match.group(1).strip()
                    # === STRIP reasoning from visible text (this was missing) ===
                    output = content.replace(match.group(0), "").strip()
                elif "reasoning_content" in raw:
                    reasoning_content = raw["reasoning_content"]
                else:
                    output = content

                    # === GROK PATH (unchanged) ===
        if provider == "grok":
            grok_resp = GrokResponse.from_dict(raw)
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
                reasoning_content=reasoning_content,
            )

        # === OLLAMA PATH (now strips reasoning when requested) ===
        if not output:                                                                    # fallback if no stripping happened
            choices = raw.get("choices", [])
            if choices and isinstance(choices[0], dict):
                output = choices[0].get("message", {}).get("content", "") or raw.get(
                    "response", ""
                )
            else:
                output = raw.get("response", "") or raw.get("message", {}).get(
                    "content", ""
                )

        return cls(
            id=raw.get("id", "ollama-unknown"),
            created_at=raw.get("created_at", 0),
            model=raw.get("model", "unknown"),
            provider=provider,
            text=output,
            tool_calls=raw.get("message", {}).get("tool_calls", [])
            or raw.get("tool_calls", []),
            usage=raw.get("usage"),
            continuation_token=raw.get("context") or continuation_token,
            raw=raw,
            extra={
                "total_duration": raw.get("total_duration"),
                "load_duration": raw.get("load_duration"),
                "eval_count": raw.get("eval_count"),
            },
            reasoning_content=reasoning_content,
        )


@dataclass(frozen=True)
class StreamingChunk:
    """
    A single token (or reasoning) chunk from a streaming LLM response.

    NEW: reasoning_delta gives you live chain-of-thought output when
    capture_reasoning=True and the model is a reasoning model (Grok-4,
    DeepSeek-R1, Qwen2.5, etc.).
    """

    delta_text: str = ""                                                                  # Normal content that goes to the final answer
    reasoning_delta: str | None = None                                                    # Live reasoning / <think> content
    finished: bool = False
    raw_chunk: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] | None = None                                                   # Only populated on the final chunk

    @property
    def has_reasoning(self) -> bool:
        """Convenience check for templates/UI."""
        return self.reasoning_delta is not None and self.reasoning_delta.strip() != ""

    def __str__(self) -> str:
        if self.has_reasoning:
            return f"Thinking: {self.reasoning_delta!r} | Text: {self.delta_text!r}"
        return self.delta_text
