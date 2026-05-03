"""
xAI-Specific Error Hierarchy (Full Backward Compatibility)

This module defines the complete xAI error taxonomy while preserving every
public name from earlier versions of the library. It delegates generic
behaviour to ``core/common/errors.py`` and adds xAI-specific subclasses
for gRPC, rate limiting, thinking mode, multimodal, caching, and batch
failures.

Design Goals
------------
- Consistent with ``ollama/errors_ollama.py`` and the shared base classes.
- No changes required for the new Git-style branching features (errors are
  orthogonal to conversation metadata).

See Also
--------
ai_api.core.common.errors
ai_api.core.ollama.errors_ollama
"""

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
    "wrap_postgres_error",
    "wrap_xai_api_error",
]


# Alias the generic base to the old xAI name (100 % backward compatible)
xAIError = AIAPIError

# Logger – re-export with original xAI names
xAILoggerError = AILoggerError
xAILoggerInitializationError = AILoggerInitializationError

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


# Wrapper aliase (kept exactly as before)
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
