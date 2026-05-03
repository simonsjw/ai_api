"""
Ollama-specific error hierarchy.

This module defines exceptions specific to the Ollama local inference server.
All exceptions inherit from ``OllamaError`` (which inherits from
``ai_api.core.common.errors.AIAPIError``), ensuring a uniform error surface
across the entire library.

High-level design
-----------------
- **Root**: ``OllamaError`` – catches everything Ollama-related.
- **API layer**: ``OllamaAPIError`` and subclasses for HTTP-level issues
  (connection refused, invalid model, malformed requests).
- **Client layer**: ``OllamaClientError`` for internal validation/state errors
  inside ``OllamaClient`` / chat helpers.
- **Local-resource layer** (Ollama-only): ``OllamaGPUMemoryWarning`` – raised
  when the local GPU runs out of VRAM or experiences memory pressure during
  inference. This error type has **no counterpart in the xAI module** because
  xAI is a remote managed service that handles GPU allocation server-side.

The 'wrap error' approach (identical to common and xAI)
-------------------------------------------------------
Every raw exception (``httpx.HTTPError``, ``asyncio.TimeoutError``,
CUDA OOM, etc.) is converted by one of the three wrapper functions:

.. code-block:: python

    from ai_api.core.ollama.errors import (
        wrap_ollama_api_error,
        wrap_ollama_gpu_mem_error,
    )

    try:
        resp = await http_client.post(...)
    except Exception as exc:
        raise wrap_ollama_api_error(exc, "Ollama chat request failed") from exc

The wrapper:
1. Creates the typed error with a human message.
2. Populates ``details={"original": "httpx.ConnectError"}``.
3. Sets ``__cause__`` so the original traceback is preserved.

Direct import pattern (no pass-through convenience importing)
-------------------------------------------------------------
Import exactly from the module that defines the name:

.. code-block:: python

    from ai_api.core.common.errors import AIAPIError, AIClientError
    from ai_api.core.ollama.errors import (
        OllamaError,
        OllamaAPIConnectionError,
        OllamaGPUMemoryWarning,
        wrap_ollama_gpu_mem_error,
    )

Never rely on re-exports or ``from .errors import *``. This makes the
provenance of every exception crystal-clear and simplifies static analysis /
IDE navigation.

Comparison with xAI errors
--------------------------
- **Ollama** (local): simpler taxonomy focused on HTTP/API + a dedicated
  GPU/memory warning (highly relevant for consumer GPUs).
- **xAI** (remote): richer set covering authentication, rate limits, gRPC
  transport, thinking-mode, multimodal, prompt-cache, and batch-specific
  failures.

Both hierarchies share the same root and wrap pattern, so
``except ai_api.core.common.errors.AIAPIError`` catches errors from either
provider (or future ones).

See Also
--------
ai_api.core.common.errors : Shared base classes, generic wrappers, and the
    canonical wrap-error contract.
ai_api.core.xai.errors : xAI counterpart with remote-service error types.
"""

from __future__ import annotations

# Direct import of shared bases (no pass-through re-export)
from ..common.errors import (
    AIAPIError,
    AIClientError,
)

__all__: list[str] = [
    "OllamaError",
    # API / transport
    "OllamaAPIError",
    "OllamaAPIConnectionError",
    "OllamaAPIInvalidRequestError",
    # Client internal
    "OllamaClientError",
    # Local resource (Ollama-specific)
    "OllamaGPUMemoryWarning",
    # Wrap helpers
    "wrap_ollama_error",
    "wrap_ollama_api_error",
    "wrap_ollama_gpu_mem_error",
]


# ----------------------------------------------------------------------
# 1. Root of all Ollama errors
# ----------------------------------------------------------------------
class OllamaError(AIAPIError):
    """Root of all Ollama-related exceptions.

    Every exception raised by ``OllamaClient``, the chat helpers, or model
    management methods inherits from this class (directly or indirectly).
    """

    pass


# ----------------------------------------------------------------------
# 2. API / HTTP transport errors (local Ollama server)
# ----------------------------------------------------------------------
class OllamaAPIError(OllamaError):
    """Errors returned by the Ollama native REST API (HTTP 4xx/5xx responses,
    model-not-found, etc.).
    """

    pass


class OllamaAPIConnectionError(OllamaAPIError):
    """Failed to establish a connection to the Ollama server.

    Typical causes: ``localhost:11434`` unreachable, server not running,
    firewall rules, or DNS issues when using a remote Ollama host.
    """

    pass


class OllamaAPIInvalidRequestError(OllamaAPIError):
    """Ollama rejected the request payload.

    Examples: unknown model name, invalid JSON schema for structured output,
    unsupported parameter combination, or context-length exceeded.
    """

    pass


# ----------------------------------------------------------------------
# 3. Client-internal errors (OllamaClient logic)
# ----------------------------------------------------------------------
class OllamaClientError(OllamaError, AIClientError):
    """Internal errors within ``OllamaClient`` or its mode-specific subclasses.

    Covers validation failures, state-machine violations, unexpected response
    shapes, and other logic errors that are not transport-related.
    """

    pass


# ----------------------------------------------------------------------
# 4. Local-resource errors – GPU / memory (Ollama-specific feature)
# ----------------------------------------------------------------------
class OllamaGPUMemoryWarning(OllamaError, AIClientError):
    """GPU memory pressure or out-of-memory condition on the local Ollama host.

    This exception is raised when the underlying inference engine (llama.cpp,
    CUDA, ROCm, etc.) cannot allocate sufficient VRAM for the requested model
    + context length + batch size.

    **This error type exists only in the Ollama module.** Remote providers
    such as xAI manage GPU resources on their infrastructure and surface
    different failure modes (rate limits, model-not-available, etc.).
    Catching ``OllamaGPUMemoryWarning`` allows applications to implement
    graceful degradation (smaller context, CPU fallback, user notification)
    that would be meaningless for cloud models.
    """

    pass


# ----------------------------------------------------------------------
# Convenience wrapper functions (Ollama-specific)
# ----------------------------------------------------------------------
def wrap_ollama_error(exc: Exception, message: str) -> OllamaError:
    """Generic wrapper for any Ollama-related exception.

    Use when the exact subcategory is unknown at catch time.

    Parameters
    ----------
    exc : Exception
        The original exception (httpx, asyncio, CUDA, etc.).
    message : str
        Human-readable description of the operation that failed.

    Returns
    -------
    OllamaError
        Typed exception with ``__cause__`` and ``details`` populated.
    """
    wrapped = OllamaError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_ollama_api_error(exc: Exception, message: str) -> OllamaAPIError:
    """Wrap errors originating from HTTP calls to the Ollama REST API.

    Parameters
    ----------
    exc : Exception
        The original exception (usually ``httpx.HTTPError`` or subclass).
    message : str
        Contextual message (e.g. "Failed to generate chat completion").

    Returns
    -------
    OllamaAPIError
    """
    wrapped = OllamaAPIError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_ollama_gpu_mem_error(exc: Exception, message: str) -> OllamaGPUMemoryWarning:
    """Wrap GPU / CUDA / ROCm out-of-memory or memory-pressure errors.

    Parameters
    ----------
    exc : Exception
        The original low-level exception (``torch.cuda.OutOfMemoryError``,
        ``RuntimeError`` with "out of memory", etc.).
    message : str
        Contextual message (e.g. "Model too large for available GPU memory").

    Returns
    -------
    OllamaGPUMemoryWarning
    """
    wrapped = OllamaGPUMemoryWarning(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped
