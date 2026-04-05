#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grok-specific custom exception hierarchy for the ai_api package.

All errors raised by GrokClient (and related components) now inherit from
GrokError. This provides clear, category-based identification of issues
while remaining fully compatible with the logger package, infopypg package,
PostgreSQL (via asyncpg), the official xAI SDK, and internal client logic.

Existing third-party errors are imported where possible and wrapped for
context (logger and infopypg define none, so we wrap the built-ins they raise).
The previous UnsupportedThinkingModeError is preserved as a specific client error.

Categories (as requested):
- Logger package
- infopypg package
- Postgres database (asyncpg)
- Grok API (xAI SDK / gRPC)
- Client (internal ai_api/GrokClient logic)

Usage in GrokClient:
    raise GrokClientError("Invalid request") from original_exc
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

    # logger and infopypg define no custom exceptions (confirmed via repo inspection)
    # → we wrap ValueError / TypeError / RuntimeError they raise

    # ----------------------------------------------------------------------
    # Base error
    # ----------------------------------------------------------------------


class GrokError(Exception):
    """Root of all GrokClient-related exceptions."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

        # ----------------------------------------------------------------------
        # 1. Logger package errors
        # ----------------------------------------------------------------------


class GrokLoggerError(GrokError):
    """Errors originating from the logger package (setup_logger, structured logging)."""

    pass


class GrokLoggerInitializationError(GrokLoggerError):
    """Logger could not be initialised (invalid config, file permissions, etc.)."""

    pass


# ----------------------------------------------------------------------
# 2. infopypg package errors
# ----------------------------------------------------------------------
class GrokInfopypgError(GrokError):
    """Errors originating from the infopypg package (settings resolution, pool management)."""

    pass


class GrokInfopypgSettingsError(GrokInfopypgError):
    """infopypg settings validation or resolution failed (validate_dict_to_ResolvedSettingsDict)."""

    pass


class GrokInfopypgPoolError(GrokInfopypgError):
    """PgPoolManager failed to acquire or maintain a connection pool."""

    pass


# ----------------------------------------------------------------------
# 3. Postgres database errors (asyncpg)
# ----------------------------------------------------------------------
class GrokPostgresError(GrokError):
    """Errors from PostgreSQL / asyncpg (connection, query execution, schema issues)."""

    pass


if asyncpg is not None:

    class GrokPostgresConnectionError(
        GrokPostgresError, asyncpg.ConnectionDoesNotExistError
    ):
        """Failed to connect to Postgres (credentials, host, SSL, etc.)."""

        pass

    class GrokPostgresQueryError(GrokPostgresError, asyncpg.PostgresError):
        """SQL execution error (syntax, constraint violation, timeout, etc.)."""

        pass
else:
    # Fallback if asyncpg not installed
    class GrokPostgresConnectionError(GrokPostgresError):
        pass

    class GrokPostgresQueryError(GrokPostgresError):
        pass

    # ----------------------------------------------------------------------
    # 4. Grok API errors (xAI SDK / gRPC)
    # ----------------------------------------------------------------------


class GrokAPIError(GrokError):
    """Errors returned by the official xAI Grok API (rate limits, model not found, etc.)."""

    pass


if grpc is not None:

    class GrokAPIConnectionError(GrokAPIError, grpc.RpcError):
        """gRPC transport / connection failure to xAI endpoints."""

        pass

    class GrokAPIAuthenticationError(GrokAPIError, grpc.RpcError):
        """Invalid or missing XAI_API_KEY."""

        pass
else:

    class GrokAPIConnectionError(GrokAPIError):
        pass

    class GrokAPIAuthenticationError(GrokAPIError):
        pass


class GrokAPIInvalidRequestError(GrokAPIError):
    """API rejected the request (bad model, malformed input, unsupported features)."""

    pass


class GrokAPIRateLimitError(GrokAPIError):
    """xAI rate limit exceeded."""

    pass


# ----------------------------------------------------------------------
# 5. Client errors (internal ai_api / GrokClient logic)
# ----------------------------------------------------------------------
class GrokClientError(GrokError):
    """Internal errors within GrokClient itself (validation, state, usage)."""

    pass


class UnsupportedThinkingModeError(GrokClientError):
    """Reasoning / 'think' mode requested on a model that does not support it."""

    pass


class GrokClientBatchError(GrokClientError):
    """Batch processing error (invalid batch_id, add_to_batch failure, etc.)."""

    pass


class GrokClientMultimodalError(GrokClientError):
    """Invalid multimodal / file attachment configuration."""

    pass


class GrokClientCacheError(GrokClientError):
    """Prompt-cache key misuse or cache-related client state error."""

    pass


# ----------------------------------------------------------------------
# Convenience factory methods (recommended usage in GrokClient)
# ----------------------------------------------------------------------


def wrap_logger_error(exc: Exception, message: str) -> GrokLoggerError:
    """Wrap any logger-originated exception."""
    wrapped = GrokLoggerError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_infopypg_error(exc: Exception, message: str) -> GrokInfopypgError:
    """Wrap any infopypg-originated exception."""
    wrapped = GrokInfopypgError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_postgres_error(exc: Exception, message: str) -> GrokPostgresError:
    """Wrap any asyncpg / Postgres exception."""
    wrapped = GrokPostgresError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_grok_api_error(exc: Exception, message: str) -> GrokAPIError:
    """Wrap any xAI SDK / gRPC exception."""
    wrapped = GrokAPIError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped
