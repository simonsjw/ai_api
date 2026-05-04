"""
Ollama-specific error hierarchy (updated with GPU memory detection).

This module defines exceptions specific to the Ollama local inference server.
All exceptions inherit from ``OllamaError`` (which inherits from
``ai_api.core.common.errors.AIAPIError``), ensuring a uniform error surface
across the entire library.

High-level design
-----------------
- **Root**: ``OllamaError`` – catches everything Ollama-related.
- **API layer**: ``OllamaAPIError`` and subclasses for HTTP-level issues.
- **Client layer**: ``OllamaClientError`` for internal validation/state errors.
- **Local-resource layer** (Ollama-only): ``OllamaGPUMemoryWarning`` – raised
  when the local GPU runs out of VRAM or experiences memory pressure during
  inference. This error type has **no counterpart in the xAI module**.

GPU Memory Availability Checks
------------------------------
We provide two mechanisms:

1. **Reactive detection** (primary, used in all inference paths):
   ``is_ollama_gpu_memory_error(exc)`` inspects the exception message for
   common CUDA/llama.cpp/ROCm OOM strings ("out of memory", "CUDA out of memory",
   "failed to allocate", "VRAM", etc.). When detected, we raise
   ``OllamaGPUMemoryWarning`` via ``wrap_ollama_gpu_mem_error``.

2. **Wrapper functions**: ``wrap_ollama_gpu_mem_error`` is now actively used
   in ``chat_turn_ollama.py``, ``chat_stream_ollama.py``, ``embeddings_ollama.py``,
   and ``chat_batch_ollama.py`` (via the turn path) so the import is never unused.

This gives callers a typed exception they can catch for graceful degradation
(switch to smaller model, reduce num_ctx, fall back to CPU, notify user, etc.).

See Also
--------
ai_api.core.common.errors
ai_api.core.xai.errors
"""

from __future__ import annotations

import re
from typing import Pattern

# Direct import of shared bases (no pass-through re-export)
from ..common.errors import (
    AIAPIError,
    AIClientError,
)

__all__: list[str] = [
    "OllamaError",
    "OllamaAPIError",
    "OllamaAPIConnectionError",
    "OllamaAPIInvalidRequestError",
    "OllamaClientError",
    "OllamaGPUMemoryWarning",
    "wrap_ollama_error",
    "wrap_ollama_api_error",
    "wrap_ollama_gpu_mem_error",
    "is_ollama_gpu_memory_error",
]


# Pre-compiled patterns for GPU memory detection (case-insensitive)
_GPU_MEM_PATTERNS: list[Pattern[str]] = [
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"cuda out of memory", re.IGNORECASE),
    re.compile(r"failed to allocate", re.IGNORECASE),
    re.compile(r"vram", re.IGNORECASE),
    re.compile(r"gpu memory", re.IGNORECASE),
    re.compile(r"insufficient memory", re.IGNORECASE),
    re.compile(r"memory pressure", re.IGNORECASE),
    re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
]


def is_ollama_gpu_memory_error(exc: Exception) -> bool:
    """Return True if the exception indicates GPU / VRAM exhaustion.

    This is the central detection function used by all Ollama inference
    helpers (chat, stream, embeddings, batch). It looks for well-known
    strings emitted by llama.cpp, CUDA, ROCm, and Ollama's HTTP error
    responses when the local GPU cannot satisfy the allocation request
    for the chosen model + context length.

    Parameters
    ----------
    exc : Exception
        Any exception raised during an Ollama HTTP call or model load.

    Returns
    -------
    bool
        True when the error is GPU-memory related and should be surfaced
        as ``OllamaGPUMemoryWarning``.
    """
    if exc is None:
        return False
    text = str(exc)
    return any(p.search(text) for p in _GPU_MEM_PATTERNS)


# ----------------------------------------------------------------------
# Root + API + Client errors (unchanged from original)
# ----------------------------------------------------------------------
class OllamaError(AIAPIError):
    """Root of all Ollama-related exceptions."""
    pass


class OllamaAPIError(OllamaError):
    """Errors returned by the Ollama native REST API."""
    pass


class OllamaAPIConnectionError(OllamaAPIError):
    """Failed to establish a connection to the Ollama server."""
    pass


class OllamaAPIInvalidRequestError(OllamaAPIError):
    """Ollama rejected the request payload."""
    pass


class OllamaClientError(OllamaError, AIClientError):
    """Internal errors within OllamaClient or its mode-specific subclasses."""
    pass


class OllamaGPUMemoryWarning(OllamaError, AIClientError):
    """GPU memory pressure or out-of-memory condition on the local Ollama host.

    Raised (via ``wrap_ollama_gpu_mem_error``) when the underlying inference
    engine cannot allocate sufficient VRAM. Callers can catch this specific
    type to implement graceful degradation (smaller context, CPU fallback,
    user notification, model switch, etc.).
    """
    pass


# ----------------------------------------------------------------------
# Wrapper functions (now actively used by inference modules)
# ----------------------------------------------------------------------
def wrap_ollama_error(exc: Exception, message: str) -> OllamaError:
    """Generic wrapper for any Ollama-related exception."""
    wrapped = OllamaError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_ollama_api_error(exc: Exception, message: str) -> OllamaAPIError:
    """Wrap errors originating from HTTP calls to the Ollama REST API."""
    wrapped = OllamaAPIError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_ollama_gpu_mem_error(exc: Exception, message: str) -> OllamaGPUMemoryWarning:
    """Wrap GPU / CUDA / ROCm out-of-memory or memory-pressure errors.

    This wrapper is now called from chat_turn_ollama.py, chat_stream_ollama.py,
    embeddings_ollama.py and the batch path whenever ``is_ollama_gpu_memory_error``
    returns True.
    """
    wrapped = OllamaGPUMemoryWarning(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped
