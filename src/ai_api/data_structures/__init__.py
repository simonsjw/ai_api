#!/usr/bin/env python3

"""
Initialization for the data_structures package.

Re-exports key symbols from submodules for convenience (e.g., from data_structures import GrokRequest).

Public API (via __all__):
- From LLM_types_openai: OPEN_AI_MESSAGE, OPEN_AI_CHOICE, OPEN_AI_USAGE, OPEN_AI_BODY,
  OPEN_AI_RESPONSE, OPEN_AI_PROMPT_OUTPUT, OPEN_AI_BATCH_OUTPUT,
- From LLM_types_grok: SaveMode, Role, GrokMessage, GrokInput, GrokRequest, GrokResponse
- From db_responses_schema: responses_default_settings, Providers, Responses
"""

from .db_responses_schema import (
    Providers,
    Responses,
)
from .LLM_types_grok import (
    GrokInput,
    GrokMessage,
    GrokRequest,
    GrokResponse,
    Role,
    SaveMode,
)
from .LLM_types_openai import (
    OPEN_AI_BATCH_OUTPUT,
    OPEN_AI_BODY,
    OPEN_AI_CHOICE,
    OPEN_AI_MESSAGE,
    OPEN_AI_PROMPT_OUTPUT,
    OPEN_AI_RESPONSE,
    OPEN_AI_USAGE,
)

__all__: list[str] = [
    "OPEN_AI_MESSAGE",
    "OPEN_AI_CHOICE",
    "OPEN_AI_USAGE",
    "OPEN_AI_BODY",
    "OPEN_AI_RESPONSE",
    "OPEN_AI_PROMPT_OUTPUT",
    "OPEN_AI_BATCH_OUTPUT",
    "SaveMode",
    "Role",
    "GrokMessage",
    "GrokInput",
    "GrokRequest",
    "GrokResponse",
    "Providers",
    "Responses",
]
