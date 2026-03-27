#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data structures package for ai_api.

This __init__.py re-exports all public dataclasses from the provider-specific
modules (grok.py and ollama.py) so users can import everything directly from
the data_structures namespace:

    from ai_api.data_structures import (
        GrokMessage, GrokInput, GrokRequest, GrokBatchRequest,
        GrokBatchResponse, GrokResponse,
        OllamaMessage, OllamaInput, OllamaRequest, OllamaResponse,
    )

The package provides a clean, provider-agnostic API while keeping each
provider's exact payload semantics (Grok's "input" vs Ollama's "messages")
encapsulated inside the respective modules. All classes are immutable dataclasses
with full validation, to_dict/from_dict helpers, and NumPy-style documentation.

Internal flow:
1. Import provider modules lazily (no circular dependencies).
2. Expose only the public classes via __all__ (prevents namespace pollution).
3. Future providers (e.g. openai.py) can be added here with a single line.

This design keeps the LLMClient (and any higher-level batch/caching logic)
completely unaware of the underlying module names while remaining efficient
(no runtime overhead beyond the initial import).

Examples
--------
Single Grok request:
    >>> from ai_api.data_structures import GrokInput, GrokRequest
    >>> inp = GrokInput.from_list([{"role": "user", "content": "Hello"}])
    >>> req = GrokRequest(input=inp)
    >>> payload = req.to_dict()

Grok batch:
    >>> from ai_api.data_structures import GrokBatchRequest
    >>> batch = GrokBatchRequest(requests=(req1, req2))
    >>> payload = batch.to_payload()

Ollama request:
    >>> from ai_api.data_structures import OllamaInput, OllamaRequest
    >>> inp = OllamaInput.from_list([{"role": "user", "content": "Hello"}])
    >>> req = OllamaRequest(input=inp)
    >>> payload = req.to_dict()
"""

from .grok import (                                                                       # Grok provider (Responses API + native batching)
    GrokBatchRequest,
    GrokBatchResponse,
    GrokInput,
    GrokMessage,
    GrokRequest,
    GrokResponse,
)
from .ollama import (                                                                     # Ollama provider (no batch support)
    OllamaInput,
    OllamaMessage,
    OllamaRequest,
    OllamaResponse,
)

__all__ = [
    # Grok classes (ordered by typical usage flow)
    "GrokMessage",
    "GrokInput",
    "GrokRequest",
    "GrokBatchRequest",
    "GrokBatchResponse",
    "GrokResponse",
    # Ollama classes (ordered by typical usage flow)
    "OllamaMessage",
    "OllamaInput",
    "OllamaRequest",
    "OllamaResponse",
]                                                                                         # Explicit __all__ for clean star-imports and IDE support.
