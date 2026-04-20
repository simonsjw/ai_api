"""
xAI-specific error hierarchy.

All original public names (xAIError, xAIAPIError, wrap_xai_api_error, etc.)
are preserved for 100 % backward compatibility with your existing xAI code.
The generic parts are now imported from the shared core/errors.py.
"""

from __future__ import annotations

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
    """Internal xAIClient errors."""

    pass


class UnsupportedThinkingModeError(xAIClientError):
    pass


class xAIClientBatchError(xAIClientError):
    pass


class xAIClientMultimodalError(xAIClientError):
    pass


class xAIClientCacheError(xAIClientError):
    pass


# Wrapper aliases (kept exactly as before)
wrap_infopypg_error = wrap_persistence_error
wrap_postgres_error = wrap_database_error


def wrap_xai_api_error(exc: Exception, message: str) -> xAIAPIError:
    """Wrap any xAI SDK / gRPC exception."""
    wrapped = xAIAPIError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped
