#!/usr/bin/env python3

"""
Initialisation for the ai_api package.

Re-exports key symbols from submodules for convenience
(e.g., from ai_api import XAIAsyncClient).

Public API (via __all__):
- Types: -
- Classes: XAIAsyncClient
"""

from .grok_client import (
    XAIAsyncClient,
)

__all__: list[str] = [
    "XAIAsyncClient",
]

#  LocalWords:  XAIAsyncClient
