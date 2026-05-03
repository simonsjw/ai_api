"""
Shared generic error hierarchy for the entire ai_api package.

This module defines a clean, provider-agnostic exception hierarchy rooted at
``AIAPIError``. It serves as the single source of truth for error types used by
all clients (Ollama, xAI, and future providers) in logging, persistence,
database operations, generic client logic, and cross-cutting concerns.

Design principles
-----------------
- **Shallow but complete taxonomy**: Only errors that are *truly common* across
  providers live here. Provider-specific concerns (remote auth/rate-limits vs.
  local GPU/memory) are defined in the per-provider modules
  ``ai_api.core.xai.errors`` and ``ai_api.core.ollama.errors``.
- **Direct imports only**: Consumers import exactly what they need from the
  module where the class is *defined*. There are no pass-through convenience
  imports or ``__init__.py`` re-exports that obscure the true location.
  Example::

      from ai_api.core.common.errors import AIClientError, wrap_client_error
      from ai_api.core.xai.errors import xAIAPIError, wrap_xai_api_error

- **Wrap-error pattern (standard across all modules)**: Every caught raw
  exception (httpx, asyncpg, grpc, CUDA OOM, etc.) is converted via a
  ``wrap_*_error(exc, message)`` helper. The helper:
    1. Instantiates the appropriate typed error with a contextual ``message``.
    2. Records ``details={"original": type(exc).__name__}`` for machine-readable
       logging / metrics.
    3. Attaches the original via ``wrapped.__cause__ = exc`` (preserves the
       full exception chain and traceback for ``raise ... from exc``).
  This pattern is implemented identically in ``common/errors.py``,
  ``xai/errors.py`` and ``ollama/errors.py`` so that every error surfaced to
  the user is a rich ``AIAPIError`` subclass.

- **Multiple inheritance for client errors**: Provider client errors inherit
  from both the provider root (``OllamaError`` / ``xAIError``) *and*
  ``AIClientError``. This allows ``except AIClientError`` to catch all internal
  client logic errors while still permitting provider-specific ``except``
  clauses.

All documentation follows the NumPy/SciPy style for consistency and
readability.

See Also
--------
ai_api.core.xai.errors : xAI-specific extensions (auth, rate limits, gRPC,
    thinking mode, multimodal, cache, batch).
ai_api.core.ollama.errors : Ollama-specific extensions (GPU/memory warnings).
"""

from __future__ import annotations

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


# ----------------------------------------------------------------------
# 1. Root of the hierarchy
# ----------------------------------------------------------------------
class AIAPIError(Exception):
    """Root of the entire ai_api error hierarchy.

    All other exceptions in the package inherit from this class, either
    directly or indirectly. It provides a uniform ``message`` attribute and an
    optional ``details`` dictionary for machine-readable context (e.g. for
    structured logging or error reporting dashboards).

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    details : dict[str, Any] or None, optional
        Additional structured information (e.g. ``{"original": "ValueError",
        "request_id": "...", "model": "grok-4"}``). Defaults to empty dict.

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
# 2. Logger errors (shared)
# ----------------------------------------------------------------------
class AILoggerError(AIAPIError):
    """Errors originating from the logger package (e.g. structlog, logging setup)."""

    pass


class AILoggerInitializationError(AILoggerError):
    """Logger could not be initialised (missing handlers, invalid config, etc.)."""

    pass


# ----------------------------------------------------------------------
# 3. Persistence / asyncpg errors (shared)
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
# 4. Database / Postgres errors (shared)
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
# 5. Generic client errors (shared)
# ----------------------------------------------------------------------
class AIClientError(AIAPIError):
    """Internal errors within any ai_api client (validation, state machine, usage).

    Provider-specific client errors (``OllamaClientError``, ``xAIClientError``)
    inherit from this class (via multiple inheritance) so that a single
    ``except AIClientError`` clause catches all client-internal failures
    regardless of provider.
    """

    pass


# ----------------------------------------------------------------------
# Convenience wrapper functions (used by all providers)
# ----------------------------------------------------------------------
def wrap_logger_error(exc: Exception, message: str) -> AILoggerError:
    """Wrap any exception as an ``AILoggerError``.

    The original exception is attached via ``__cause__`` and its type is
    recorded in ``details["original"]``. This is the canonical pattern used
    throughout the codebase.

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
    """Wrap any exception as an ``AIPersistenceError``.

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
    """Wrap any exception as an ``AIDatabaseError``.

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
    """Wrap any exception as an ``AIClientError``.

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
