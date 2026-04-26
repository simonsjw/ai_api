"""
xAI-specific error hierarchy with full backward compatibility.

This module preserves every original public name from earlier versions of the
library (``xAIError``, ``xAIAPIError``, ``wrap_xai_api_error``, etc.) while
delegating the generic parts to the shared ``core/common/errors.py``.

High-level view of xAI error handling
-------------------------------------
- Re-exports generic errors under their historical xAI names
  (e.g. ``xAIInfopypgError`` → ``AIPersistenceError``).
- Adds xAI-specific API errors that inherit from ``xAIAPIError`` (which
  inherits from ``AIAPIError``).
- Includes gRPC-aware subclasses when the ``grpc`` package is available
  (xAI uses gRPC under the hood for some transports).
- Client-specific errors cover remote-service concerns: thinking mode,
  multimodal inputs, caching, batch failures, rate limits, authentication.

Comparison with Ollama errors
-----------------------------
- xAI: rich taxonomy reflecting a managed remote service (gRPC, rate limits,
  auth, thinking/multimodal/cache-specific errors, batch errors).
- Ollama: simpler set focused on local HTTP + a dedicated GPU-memory warning
  (much more relevant for local deployments).
- Both inherit from the same common base, so ``except AIAPIError`` catches
  everything from either provider.

All wrappers set ``__cause__`` and populate a ``details`` dict for excellent
debuggability and logging.

See Also
--------
ai_api.core.common.errors
    The canonical generic error classes and wrappers.
ai_api.core.ollama.errors_ollama
    The simpler Ollama counterpart.
"""

import asyncio
from typing import Any

try:
    import asyncpg
except ImportError:
    asyncpg = None                                                                        # type: ignore[assignment]

try:
    import grpc
except ImportError:
    grpc = None                                                                           # type: ignore[assignment]

# NEW: import shared generic base
from ..common.errors import (
    AIAPIError,
    AIClientError,
    AIDatabaseConnectionError,
    AIDatabaseError,
    AIDatabaseQueryError,
    AILoggerError,
    AILoggerInitializationError,
    AIPersistenceError,
    AIPersistencePoolError,
    AIPersistenceSettingsError,
    wrap_client_error,
    wrap_database_error,
    wrap_logger_error,
    wrap_persistence_error,
)

__all__: list[str] = [
    # Base (kept for backward compatibility)
    "xAIError",
    # Logger (re-export original names)
    "xAILoggerError",
    "xAILoggerInitializationError",
    # Persistence (re-export original names)
    "xAIInfopypgError",
    "xAIInfopypgSettingsError",
    "xAIInfopypgPoolError",
    # Database
    "xAIPostgresError",
    "xAIPostgresConnectionError",
    "xAIPostgresQueryError",
    # xAI API / gRPC
    "xAIAPIError",
    "xAIAPIConnectionError",
    "xAIAPIAuthenticationError",
    "xAIAPIInvalidRequestError",
    "xAIAPIRateLimitError",
    # Client
    "xAIClientError",
    "UnsupportedThinkingModeError",
    "xAIClientBatchError",
    "xAIClientMultimodalError",
    "xAIClientCacheError",
    # Wrappers (kept exactly as before)
    "wrap_logger_error",
    "wrap_infopypg_error",
    "wrap_postgres_error",
    "wrap_xai_api_error",
]


# Alias the generic base to the old xAI name (100 % backward compatible)
xAIError = AIAPIError

# Logger – re-export with original xAI names
xAILoggerError = AILoggerError
xAILoggerInitializationError = AILoggerInitializationError

# Persistence – re-export with original xAI names
xAIInfopypgError = AIPersistenceError
xAIInfopypgSettingsError = AIPersistenceSettingsError
xAIInfopypgPoolError = AIPersistencePoolError

# Database – re-export with original xAI names
xAIPostgresError = AIDatabaseError
xAIPostgresConnectionError = AIDatabaseConnectionError
xAIPostgresQueryError = AIDatabaseQueryError


# xAI-specific API errors (these stay in this file)
class xAIAPIError(AIAPIError):
    """Errors returned by the official xAI API / gRPC."""

    pass


if grpc is not None:

    class xAIAPIConnectionError(xAIAPIError, grpc.RpcError):
        pass

    class xAIAPIAuthenticationError(xAIAPIError, grpc.RpcError):
        pass
else:

    class xAIAPIConnectionError(xAIAPIError):
        pass

    class xAIAPIAuthenticationError(xAIAPIError):
        pass


class xAIAPIInvalidRequestError(xAIAPIError):
    pass


class xAIAPIRateLimitError(xAIAPIError):
    pass


# xAI-specific client errors
class xAIClientError(AIClientError):
    """Internal xAIClient errors (validation, state, usage)."""

    pass


class UnsupportedThinkingModeError(xAIClientError):
    """Requested thinking mode is not supported by the current model or SDK version."""

    pass


class xAIClientBatchError(xAIClientError):
    """Error specific to batch processing (e.g. mismatched response_model list length)."""

    pass


class xAIClientMultimodalError(xAIClientError):
    """Error related to multimodal (image/video) input handling."""

    pass


class xAIClientCacheError(xAIClientError):
    """Error related to prompt caching or context caching features."""

    pass


# Wrapper aliases (kept exactly as before)
wrap_infopypg_error = wrap_persistence_error
wrap_postgres_error = wrap_database_error


def wrap_xai_api_error(exc: Exception, message: str) -> xAIAPIError:
    """Wrap any xAI SDK / gRPC exception.

    Parameters
    ----------
    exc : Exception
        The original exception (often a gRPC or httpx error).
    message : str
        Contextual message describing the operation that failed.

    Returns
    -------
    xAIAPIError
        Properly typed exception with ``__cause__`` and ``details``.
    """
    wrapped = xAIAPIError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped
