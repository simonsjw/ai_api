"""
xAI-specific custom exception hierarchy for the ai_api package.

All errors raised by xAIClient (and related components) now inherit from
xAIError. This provides clear, category-based identification of issues
while remaining fully compatible with the logger package, infopypg package,
PostgreSQL (via asyncpg), the official xAI SDK, and internal client logic.

Existing third-party errors are imported where possible and wrapped for
context (logger and infopypg define none, so we wrap the built-ins they raise).
The previous UnsupportedThinkingModeError is preserved as a specific client error.

Categories (as requested):
- Logger package
- infopypg package
- Postgres database (asyncpg)
- xAI API (xAI SDK / gRPC)
- Client (internal ai_api/xAIClient logic)

Usage in xAIClient:
    raise xAIClientError("Invalid request") from original_exc
    # or the convenience methods below
"""

from __future__ import annotations

import asyncio
from typing import Any

# ----------------------------------------------------------------------
# Third-party imports (wrapped where they exist)
# ----------------------------------------------------------------------
try:
    import asyncpg                                                                        # postgres
except ImportError:
    asyncpg = None                                                                        # type: ignore[assignment]

try:
    import grpc                                                                           # xAI SDK is gRPC-based
except ImportError:
    grpc = None                                                                           # type: ignore[assignment]


__all__: list[str] = [
    # Base exception
    "xAIError",
    # Logger package errors
    "xAILoggerError",
    "xAILoggerInitializationError",
    # infopypg package errors
    "xAIInfopypgError",
    "xAIInfopypgSettingsError",
    "xAIInfopypgPoolError",
    # Postgres / asyncpg errors
    "xAIPostgresError",
    "xAIPostgresConnectionError",
    "xAIPostgresQueryError",
    # xAI API / gRPC errors
    "xAIAPIError",
    "xAIAPIConnectionError",
    "xAIAPIAuthenticationError",
    "xAIAPIInvalidRequestError",
    "xAIAPIRateLimitError",
    # Client errors
    "xAIClientError",
    "UnsupportedThinkingModeError",
    "xAIClientBatchError",
    "xAIClientMultimodalError",
    "xAIClientCacheError",
    # Convenience wrapper functions
    "wrap_logger_error",
    "wrap_infopypg_error",
    "wrap_postgres_error",
    "wrap_xai_api_error",
]


# logger and infopypg define no custom exceptions (confirmed via repo inspection)
# → we wrap ValueError / TypeError / RuntimeError they raise

# ----------------------------------------------------------------------
# Base error
# ----------------------------------------------------------------------


class xAIError(Exception):
    """Root of all xAIClient-related exceptions."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

        # ----------------------------------------------------------------------
        # 1. Logger package errors
        # ----------------------------------------------------------------------


class xAILoggerError(xAIError):
    """Errors originating from the logger package (setup_logger, structured logging)."""

    pass


class xAILoggerInitializationError(xAILoggerError):
    """Logger could not be initialised (invalid config, file permissions, etc.)."""

    pass


# ----------------------------------------------------------------------
# 2. infopypg package errors
# ----------------------------------------------------------------------
class xAIInfopypgError(xAIError):
    """Errors originating from the infopypg package (settings resolution, pool management)."""

    pass


class xAIInfopypgSettingsError(xAIInfopypgError):
    """infopypg settings validation or resolution failed (validate_dict_to_ResolvedSettingsDict)."""

    pass


class xAIInfopypgPoolError(xAIInfopypgError):
    """PgPoolManager failed to acquire or maintain a connection pool."""

    pass


# ----------------------------------------------------------------------
# 3. Postgres database errors (asyncpg)
# ----------------------------------------------------------------------
class xAIPostgresError(xAIError):
    """Errors from PostgreSQL / asyncpg (connection, query execution, schema issues)."""

    pass


if asyncpg is not None:

    class xAIPostgresConnectionError(
        xAIPostgresError, asyncpg.ConnectionDoesNotExistError
    ):
        """Failed to connect to Postgres (credentials, host, SSL, etc.)."""

        pass

    class xAIPostgresQueryError(xAIPostgresError, asyncpg.PostgresError):
        """SQL execution error (syntax, constraint violation, timeout, etc.)."""

        pass
else:
    # Fallback if asyncpg not installed
    class xAIPostgresConnectionError(xAIPostgresError):
        pass

    class xAIPostgresQueryError(xAIPostgresError):
        pass

    # ----------------------------------------------------------------------
    # 4. xAI API errors (xAI SDK / gRPC)
    # ----------------------------------------------------------------------


class xAIAPIError(xAIError):
    """Errors returned by the official xAI xAI API (rate limits, model not found, etc.)."""

    pass


if grpc is not None:

    class xAIAPIConnectionError(xAIAPIError, grpc.RpcError):
        """gRPC transport / connection failure to xAI endpoints."""

        pass

    class xAIAPIAuthenticationError(xAIAPIError, grpc.RpcError):
        """Invalid or missing XAI_API_KEY."""

        pass
else:

    class xAIAPIConnectionError(xAIAPIError):
        pass

    class xAIAPIAuthenticationError(xAIAPIError):
        pass


class xAIAPIInvalidRequestError(xAIAPIError):
    """API rejected the request (bad model, malformed input, unsupported features)."""

    pass


class xAIAPIRateLimitError(xAIAPIError):
    """xAI rate limit exceeded."""

    pass


# ----------------------------------------------------------------------
# 5. Client errors (internal ai_api / xAIClient logic)
# ----------------------------------------------------------------------
class xAIClientError(xAIError):
    """Internal errors within xAIClient itself (validation, state, usage)."""

    pass


class UnsupportedThinkingModeError(xAIClientError):
    """Reasoning / 'think' mode requested on a model that does not support it."""

    pass


class xAIClientBatchError(xAIClientError):
    """Batch processing error (invalid batch_id, add_to_batch failure, etc.)."""

    pass


class xAIClientMultimodalError(xAIClientError):
    """Invalid multimodal / file attachment configuration."""

    pass


class xAIClientCacheError(xAIClientError):
    """Prompt-cache key misuse or cache-related client state error."""

    pass


# ----------------------------------------------------------------------
# Convenience factory methods (recommended usage in xAIClient)
# ----------------------------------------------------------------------


def wrap_logger_error(exc: Exception, message: str) -> xAILoggerError:
    """Wrap any logger-originated exception."""
    wrapped = xAILoggerError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_infopypg_error(exc: Exception, message: str) -> xAIInfopypgError:
    """Wrap any infopypg-originated exception."""
    wrapped = xAIInfopypgError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_postgres_error(exc: Exception, message: str) -> xAIPostgresError:
    """Wrap any asyncpg / Postgres exception."""
    wrapped = xAIPostgresError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_xai_api_error(exc: Exception, message: str) -> xAIAPIError:
    """Wrap any xAI SDK / gRPC exception."""
    wrapped = xAIAPIError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped
