"""
Ollama-specific error hierarchy.

Mirrors the exact style of errors_xai.py for consistency.
Reuses the shared generic base from ../errors.py.
"""

from __future__ import annotations

from typing import Any

# Import shared generic base
from ..common.errors import (
    AIAPIError,
    AIClientError,
    AIPersistenceError,
    wrap_client_error,
    wrap_persistence_error,
)

__all__: list[str] = [
    "OllamaError",
    "OllamaAPIError",
    "OllamaAPIConnectionError",
    "OllamaAPIInvalidRequestError",
    "OllamaClientError",
    "wrap_ollama_error",
    "wrap_ollama_api_error",
]


class OllamaError(AIAPIError):
    """Root of all Ollama-related exceptions."""

    pass


class OllamaAPIError(OllamaError):
    """Errors returned by the Ollama native API (HTTP 4xx/5xx, model not found, etc.)."""

    pass


class OllamaAPIConnectionError(OllamaAPIError):
    """Failed to connect to Ollama server (localhost:11434 unreachable, etc.)."""

    pass


class OllamaAPIInvalidRequestError(OllamaAPIError):
    """Ollama rejected the request (invalid model, malformed format, etc.)."""

    pass


class OllamaClientError(OllamaError, AIClientError):
    """Internal errors within OllamaClient."""

    pass


# Convenience wrappers (used by chat_*_ollama.py and ollama_client.py)
def wrap_ollama_error(exc: Exception, message: str) -> OllamaError:
    """Generic wrapper for any Ollama-related exception."""
    wrapped = OllamaError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_ollama_api_error(exc: Exception, message: str) -> OllamaAPIError:
    """Wrap errors from httpx calls to Ollama."""
    wrapped = OllamaAPIError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped
