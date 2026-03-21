#!/usr/bin/env python3
"""
Unified request abstraction for all LLM providers.

This module defines the single immutable `LLMRequest` dataclass that replaces
the provider-specific `GrokRequest`. It centralises every parameter that is
common across Grok and Ollama while providing two extension dictionaries
(`provider_options` and `backend_options`) for anything unique to a
particular back-end. A dedicated `sys_spec` field captures hardware-level
details (quantisation, VRAM requirements, etc.) for future resource
management modules.

Design priorities (in order):
1. Efficiency — single frozen dataclass, minimal memory, fast `to_dict`.
2. Clarity — explicit separation of universal vs provider-specific fields.
3. Readability — short methods, full NumPy-style docstrings, inline
   comments after column 90.

All existing callers that used `GrokRequest` will migrate to `LLMRequest`
with zero behaviour change for Grok (the factory maps automatically).

Future providers require only a new concrete client; no changes to this file.
"""

from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel

from ..data_structures.LLM_types_grok import GrokInput                                    # reuse existing validated input


@dataclass(frozen=True)
class LLMRequest:
    """
    Immutable request for any LLM provider (Grok, Ollama, future).

    Parameters
    ----------
    input : GrokInput
        The conversation history (reuses existing immutable GrokInput).
    model : str
        Model identifier (e.g. "grok-3", "qwen3-coder-next:latest").
    temperature : float | None, optional
        Sampling temperature (0.0-2.0). None = provider default.
    top_p : float | None, optional
        Nucleus sampling (0.0-1.0). None = provider default.
    max_output_tokens : int | None, optional
        Hard limit on generated tokens.
    tools : list[dict[str, Any]] | None, optional
        Tool definitions (OpenAI-compatible schema).
    tool_choice : str | dict[str, Any] | None, optional
        Tool-calling strategy ("auto", "none", "required", …).
    structured_schema : dict[str, Any] | type[BaseModel] | None, optional
        JSON schema or Pydantic model for structured output.
    seed : int | None, optional
        Random seed for reproducible outputs (supported by both Grok and
        Ollama).
    provider_options : dict[str, Any] | None, optional
        Grok-specific fields (store, previous_response_id, …).
    backend_options: dict[str, Any] | OllamaOptions | None = None
        # Ollama-specific fields (num_ctx, num_gpu, etc.). Use OllamaOptions for type safety.
    sys_spec : dict[str, Any] | None, optional
        Hardware-level metadata (quantisation, expected_vram_gb,
        min_spare_vram_gb). Saved in persistence meta JSONB; used later
        by resource-estimation module.
    capture_reasoning: bool, default false.

    Notes
    -----
    - Frozen for thread-safety and hashability.
    - `to_dict` produces a minimal payload suitable for any concrete client.
    - Structured schema is automatically converted to the correct
      `response_format` key.
    - capture_reasoning - If True, request and extract reasoning/thinking traces
      when the model provides them.
      Grok-4: adds include=["reasoning.encrypted_content"]
      Ollama: parses <think> tags or reasoning_content field.
      Default False to reduce token usage.
    """

    input: GrokInput
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    structured_schema: dict[str, Any] | type[BaseModel] | None = None
    seed: int | None = None
    provider_options: dict[str, Any] | None = None
    backend_options: dict[str, Any] | None = None
    sys_spec: dict[str, Any] | None = None
    capture_reasoning: bool = False

    def to_dict(self) -> dict[str, Any]:
        """
        Convert request to minimal dictionary for API submission.

        Returns
        -------
        dict[str, Any]
            JSON-serialisable payload (None values omitted).

        Flow
        ----
        1. Build core fields from immutable attributes.
        2. Add structured_schema if present (via private helper).
        3. Merge provider_options and backend_options.
        4. Filter None values (keeps payload tiny).
        """
        base: dict[str, Any] = {                                                          # efficient single-pass dict
            "input": self.input.to_list(),
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "seed": self.seed,
        }

        if self.structured_schema is not None:                                            # structured output support
            base["response_format"] = {
                "type": "json_object",
                "schema": self._serialise_schema(),
            }

            # merge extensions (last-wins for any overlap)
        if self.provider_options:
            base.update(self.provider_options)
        if self.backend_options:
            base.update(self.backend_options)

        return {
            k: v for k, v in base.items() if v is not None
        }                                                                                 # omit None for clean JSON

    def _serialise_schema(self) -> dict[str, Any]:
        """
        Private helper to convert structured_schema to JSON schema.

        Returns
        -------
        dict[str, Any]
            Ready-to-use schema dict.

        Raises
        ------
        ValueError
            If schema is neither dict nor Pydantic BaseModel subclass.
        """
        if isinstance(self.structured_schema, dict):
            return self.structured_schema
        if isinstance(self.structured_schema, type) and issubclass(
            self.structured_schema, BaseModel
        ):
            # Explicit cast for strict type checkers like Pyrefly.
            # After the guard we know this is safe; cast just satisfies static analysis.
            model_cls: type[BaseModel] = cast(type[BaseModel], self.structured_schema)
            return model_cls.model_json_schema()
        raise ValueError("structured_schema must be dict or BaseModel subclass.")
