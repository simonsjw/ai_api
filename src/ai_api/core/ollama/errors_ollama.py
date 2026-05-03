"""
Ollama-specific error hierarchy.

All exceptions inherit from ``ai_api.core.common.errors.AIAPIError``,
ensuring consistent error handling across providers and across
branching / non-branching code paths.

High-level view
---------------
- All Ollama errors inherit from ``OllamaError`` (which inherits from
  ``AIAPIError``).
- API-level errors (connection, invalid request) are separated from
  client-internal errors.
- A dedicated ``OllamaGPUMemoryWarning`` exists because local Ollama
  deployments frequently hit GPU memory limits — this is much less common
  with the remote xAI service.

Comparison with xAI errors
--------------------------
- Ollama: simpler taxonomy focused on local HTTP/API issues + GPU memory.
- xAI: much richer set (gRPC, authentication, rate limits, thinking mode,
  multimodal, cache, batch-specific errors) because it is a remote managed
  service with more failure modes.

All wrapper functions follow the same pattern as the common wrappers so that
every caught exception is turned into a properly typed, context-rich error
with ``__cause__`` set.

See Also
--------
ai_api.core.common.errors.AIAPIError : root of the hierarchy.
ai_api.core.common.chat_session : ``ChatSession`` surfaces these errors
    unchanged (persistence or branching failures are wrapped only when
    they originate inside ``persist_chat_turn``).
"""

from __future__ import annotations

# Import shared generic base
from ..common.errors import (
    AIAPIError,                                                                           # Root of the entire ai_api error hierarchy. Error message and context dict provided.
    AIClientError,                                                                        # Generic internal errors within any ai_api client (validation, state machine, usage).
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
    """Internal errors within OllamaClient (validation, state, usage)."""

    pass


class OllamaGPUMemoryWarning(OllamaError, AIClientError):
    """GPU memory pressure or out-of-memory condition on the local Ollama host."""

    pass


# Convenience wrappers (used by chat_*_ollama.py and ollama_client.py)
def wrap_ollama_error(exc: Exception, message: str) -> OllamaError:
    """Generic wrapper for any Ollama-related exception.

    Parameters
    ----------
    exc : Exception
        The original exception.
    message : str
        Contextual message.

    Returns
    -------
    OllamaError
    """
    wrapped = OllamaError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_ollama_api_error(exc: Exception, message: str) -> OllamaAPIError:
    """Wrap errors from httpx calls to Ollama.

    Parameters
    ----------
    exc : Exception
        The original exception (usually httpx.HTTPError).
    message : str
        Contextual message.

    Returns
    -------
    OllamaAPIError
    """
    wrapped = OllamaAPIError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_ollama_gpu_mem_error(exc: Exception, message: str) -> OllamaGPUMemoryWarning:
    """Wrap errors from GPU memory issues in Ollama.

    Parameters
    ----------
    exc : Exception
        The original exception (usually CUDA / torch OOM).
    message : str
        Contextual message.

    Returns
    -------
    OllamaGPUMemoryWarning
    """
    wrapped = OllamaGPUMemoryWarning(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped
