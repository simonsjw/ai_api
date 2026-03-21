#!/usr/bin/env python3

"""
Initialisation for the core package.

Re-exports the unified request and response abstractions so they can be
imported conveniently from the top level of ai_api:

    from ai_api.core import LLMRequest, LLMResponse, StreamingChunk

These three classes form the single, provider-agnostic contract used by
both GrokConcreteClient and OllamaConcreteClient (and any future
providers). They replace the original Grok-specific dataclasses while
preserving full backwards compatibility through the factory.

Public API (via __all__):
- LLMRequest          — immutable request with sys_spec, seed, and
                        provider/backend extension dicts
- LLMResponse         — immutable response with continuation_token
                        abstraction (Grok conv_id or Ollama context)
- StreamingChunk      — single token or tool-call delta for the
                        streaming interface

All persistence, logging, and dashboard extraction layers now operate
exclusively on these types.
"""

from .request import LLMRequest
from .response import LLMResponse, StreamingChunk

__all__: list[str] = [
    "LLMRequest",
    "LLMResponse",
    "StreamingChunk",
]
