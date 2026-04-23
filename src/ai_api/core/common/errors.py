"""
Shared generic error hierarchy for the entire ai_api package.

This module defines a clean, provider-agnostic exception hierarchy rooted at
`AIAPIError`. It serves as the single source of truth for error types used by
all clients (Ollama, xAI, and future providers) in logging, persistence,
database operations, and generic client logic.

What it does
------------
- Provides a shallow but complete taxonomy of errors that are truly common
  across providers.
- Supplies thin wrapper functions (`wrap_*_error`) that attach the original
  exception as `__cause__` and enrich it with a `details` dict. This enables
  rich logging and debugging without losing the original traceback.

How it does it
--------------
- All exceptions inherit from `AIAPIError`, which stores `message` and
  optional `details`.
- Provider-specific errors (transport errors, model-specific quirks, API
  rate limits, etc.) live in the per-provider `errors_*.py` modules and
  inherit from these base classes (see `errors_xai.py` and `errors_ollama.py`).
- The wrappers are convenience functions used throughout the codebase so that
  every caught exception is turned into a properly typed `AIAPIError`
  subclass with context.

Examples — usage by clients
---------------------------
Typical pattern in any chat / embedding / persistence code:

.. code-block:: python

    from ai_api.core.common.errors import (
        wrap_client_error,
        wrap_persistence_error,
        AIClientError,
    )

    try:
        response = await client.create_chat(...)
    except Exception as exc:
        raise wrap_client_error(exc, "Chat creation failed") from exc

    # Or for persistence (used inside PersistenceManager and chat modules)
    try:
        rid, ts = await persistence_manager.persist_request(request)
    except Exception as exc:
        raise wrap_persistence_error(exc, "Failed to persist request") from exc

    # Catching specific types
    try:
        ...
    except AIClientError as e:
        logger.error("Client error", extra={"details": e.details})

See Also
--------
ai_api.core.ollama.errors_ollama, ai_api.core.xai.errors_xai
    Provider-specific error subclasses that inherit from the classes defined here.
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
    # Persistence (asyncpg / Postgres)
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
    """Root of the entire ai_api error hierarchy.

    All other exceptions in the package inherit from this class, either
    directly or indirectly. It provides a uniform `message` attribute and an
    optional `details` dictionary for machine-readable context.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    details : dict[str, Any] or None, optional
        Additional structured information (e.g. {"original": "ValueError",
        "request_id": "..."}). Defaults to empty dict.

    Attributes
    ----------
    message : str
        The error message.
    details : dict[str, Any]
        Extra context provided at construction time.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

        # ----------------------------------------------------------------------
        # 1. Logger errors (shared)
        # ----------------------------------------------------------------------


class AILoggerError(AIAPIError):
    """Errors originating from the logger package (e.g. structlog, logging setup)."""

    pass


class AILoggerInitializationError(AILoggerError):
    """Logger could not be initialised (missing handlers, invalid config, etc.)."""

    pass


# ----------------------------------------------------------------------
# 2. Persistence / asyncpg errors (shared)
# ----------------------------------------------------------------------
class AIPersistenceError(AIAPIError):
    """Errors originating from the persistence layer (PersistenceManager, asyncpg)."""

    pass


class AIPersistenceSettingsError(AIPersistenceError):
    """Settings resolution or validation failed (bad db_url, missing json_dir, etc.)."""

    pass


class AIPersistencePoolError(AIPersistenceError):
    """Failed to acquire or maintain a connection pool to Postgres."""

    pass


# ----------------------------------------------------------------------
# 3. Database / Postgres errors (shared)
# ----------------------------------------------------------------------
class AIDatabaseError(AIAPIError):
    """Errors from PostgreSQL / asyncpg operations."""

    pass


class AIDatabaseConnectionError(AIDatabaseError):
    """Failed to connect to or acquire a connection from Postgres."""

    pass


class AIDatabaseQueryError(AIDatabaseError):
    """SQL execution error (syntax, constraint violation, timeout, etc.)."""

    pass


# ----------------------------------------------------------------------
# 4. Generic client errors (shared)
# ----------------------------------------------------------------------
class AIClientError(AIAPIError):
    """Internal errors within any ai_api client (validation, state machine, usage)."""

    pass


# ----------------------------------------------------------------------
# Convenience wrapper functions (used by all providers)
# ----------------------------------------------------------------------
def wrap_logger_error(exc: Exception, message: str) -> AILoggerError:
    """Wrap any exception as an AILoggerError.

    The original exception is attached via `__cause__` and its type is
    recorded in `details["original"]`.

    Parameters
    ----------
    exc : Exception
        The original exception that occurred.
    message : str
        Contextual message describing what the client was trying to do.

    Returns
    -------
    AILoggerError
        A new exception ready to be raised or logged.
    """
    wrapped = AILoggerError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_persistence_error(exc: Exception, message: str) -> AIPersistenceError:
    """Wrap any exception as an AIPersistenceError.

    Parameters
    ----------
    exc : Exception
        The original exception.
    message : str
        Contextual message.

    Returns
    -------
    AIPersistenceError
    """
    wrapped = AIPersistenceError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_database_error(exc: Exception, message: str) -> AIDatabaseError:
    """Wrap any exception as an AIDatabaseError.

    Parameters
    ----------
    exc : Exception
        The original exception.
    message : str
        Contextual message.

    Returns
    -------
    AIDatabaseError
    """
    wrapped = AIDatabaseError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_client_error(exc: Exception, message: str) -> AIClientError:
    """Wrap any exception as an AIClientError.

    Parameters
    ----------
    exc : Exception
        The original exception.
    message : str
        Contextual message (e.g. "Failed to create chat session").

    Returns
    -------
    AIClientError
    """
    wrapped = AIClientError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped
