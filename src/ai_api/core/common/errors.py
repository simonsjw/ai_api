"""
Shared generic error hierarchy for the entire ai_api package.

This is the common foundation for ALL providers (xAI, Ollama, future OpenAI, etc.).
It contains only errors that are truly provider-agnostic:

- Logger
- Persistence / infopypg
- Database / Postgres
- General client errors

Provider-specific errors (API transport, model quirks, etc.) live in the per-provider
errors_*.py files and inherit from these base classes.

All original xAI error names are preserved via the thin wrapper in errors_xai.py.
"""

from __future__ import annotations

import asyncio
from typing import Any

__all__: list[str] = [
    # Base
    "AIAPIError",
    # Logger
    "AILoggerError",
    "AILoggerInitializationError",
    # Persistence (infopypg)
    "AIPersistenceError",
    "AIPersistenceSettingsError",
    "AIPersistencePoolError",
    # Database / Postgres
    "AIDatabaseError",
    "AIDatabaseConnectionError",
    "AIDatabaseQueryError",
    # Generic client errors
    "AIClientError",
    # Convenience wrappers (used by all providers)
    "wrap_logger_error",
    "wrap_persistence_error",
    "wrap_database_error",
    "wrap_client_error",
]


class AIAPIError(Exception):
    """Root of the entire ai_api error hierarchy."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

        # ----------------------------------------------------------------------
        # 1. Logger errors (shared)
        # ----------------------------------------------------------------------


class AILoggerError(AIAPIError):
    """Errors originating from the logger package."""

    pass


class AILoggerInitializationError(AILoggerError):
    """Logger could not be initialised."""

    pass


# ----------------------------------------------------------------------
# 2. Persistence / infopypg errors (shared)
# ----------------------------------------------------------------------
class AIPersistenceError(AIAPIError):
    """Errors originating from persistence layer (infopypg, PgPoolManager, etc.)."""

    pass


class AIPersistenceSettingsError(AIPersistenceError):
    """Settings resolution or validation failed."""

    pass


class AIPersistencePoolError(AIPersistenceError):
    """Failed to acquire or maintain a connection pool."""

    pass


# ----------------------------------------------------------------------
# 3. Database / Postgres errors (shared)
# ----------------------------------------------------------------------
class AIDatabaseError(AIAPIError):
    """Errors from PostgreSQL / asyncpg."""

    pass


class AIDatabaseConnectionError(AIDatabaseError):
    """Failed to connect to Postgres."""

    pass


class AIDatabaseQueryError(AIDatabaseError):
    """SQL execution error."""

    pass


# ----------------------------------------------------------------------
# 4. Generic client errors (shared)
# ----------------------------------------------------------------------
class AIClientError(AIAPIError):
    """Internal errors within any ai_api client (validation, state, usage)."""

    pass


# ----------------------------------------------------------------------
# Convenience wrapper functions (used by all providers)
# ----------------------------------------------------------------------
def wrap_logger_error(exc: Exception, message: str) -> AILoggerError:
    wrapped = AILoggerError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_persistence_error(exc: Exception, message: str) -> AIPersistenceError:
    wrapped = AIPersistenceError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_database_error(exc: Exception, message: str) -> AIDatabaseError:
    wrapped = AIDatabaseError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_client_error(exc: Exception, message: str) -> AIClientError:
    wrapped = AIClientError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped
