#!/usr/bin/env python3
"""
Core type aliases for the unified LLM API.

All type aliases that are shared across request.py, response.py,
streaming.py, base.py, and the concrete clients live here.
This eliminates duplication and makes adding future providers trivial.

Design note: Only lightweight type aliases (no dataclasses or heavy logic).
"""

from __future__ import annotations

from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────────────
# Public types used everywhere
# ─────────────────────────────────────────────────────────────────────────────
type ProviderLiteral = Literal["grok", "ollama"]
"""
The only two back-ends currently supported.
Adding a new provider (e.g. "anthropic") only requires updating this line
and the factory.
"""

type ContinuationToken = str | list[int] | bytes | None
"""
Opaque continuation token that unifies conversation caching:
- Grok  → str (x-grok-conv-id header)
- Ollama → list[int] (native context array)
- Future providers can use bytes or whatever they need.
"""

type SaveMode = Literal["none", "json_files", "postgres"]
"""
Re-exported here for convenience (originally in data_structures).
Having it in core/types.py makes the whole package import cleaner:
    from ai_api.core.types import SaveMode
"""

type JSONSchema = dict[str, Any]
"""
Any JSON schema (used by structured_schema in LLMRequest).
"""

# Future types you will almost certainly want:
# type ToolDefinition = dict[str, Any]
# type StructuredSchema = dict[str, Any] | type[BaseModel]
