#!/usr/bin/env python3

"""
Initialisation for the ai_api package.

This is the single public entry point for the entire library. It
re-exports the unified factory and core types so users can start
working immediately with both remote (Grok) and local (Ollama)
models using identical code.

Recommended usage:

    from ai_api import create
    from ai_api.core import LLMRequest

    client = await create(
        provider="ollama",          # or "grok"
        model="qwen3-coder-next:latest",
        org="localhost",            # only for Ollama
    )

    req = LLMRequest(
        input=your_grok_input,
        model=client.model,         # or any valid model name
        temperature=0.7,
        seed=42,
    )

    results = await client.submit_batch([req])
    # or
    async for chunk in client.stream(req):
        print(chunk.delta_text)

Public API (via __all__):
- create            — async factory returning a ready BaseAsyncProviderClient
- LLMRequest        — immutable unified request (with sys_spec, seed, etc.)
- LLMResponse       — immutable unified response (with continuation_token)
- StreamingChunk    — single token or tool-call delta for streaming

All persistence, logging, concurrency, and error handling are handled
transparently by the concrete clients. No provider-specific code is
required in user applications.
"""

from .core import LLMRequest, LLMResponse, StreamingChunk
from .factory import create

__all__: list[str] = [
    "create",
    "LLMRequest",
    "LLMResponse",
    "StreamingChunk",
]

#  LocalWords:  create LLMRequest LLMResponse StreamingChunk
