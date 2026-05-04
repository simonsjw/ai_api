"""
Shared generic error hierarchy for the entire ai_api package.

This module defines a clean, provider-agnostic exception hierarchy rooted at
``APIError``. It is the single source of truth for error types used by all
clients (Ollama, xAI, and future providers), the persistence layer, logging
setup, and cross-cutting concerns.

All wrapping of low-level exceptions should be performed exclusively through
the generic ``wrap_error`` factory. Thin convenience wrappers have been
removed in favour of the single, type-safe generic.

Design principles
-----------------
- **Single root**: Every exception ultimately inherits from ``APIError``.
  This allows ``except APIError`` to catch everything while still permitting
  fine-grained ``except`` clauses.
- **Shallow taxonomy**: Only truly common categories live here.
  Provider-specific concerns belong in ``ai_api.core.xai.errors_xai`` and
  ``ai_api.core.ollama.errors_ollama``.
- **Wrap-error pattern**: All raw exceptions are converted via the single
  ``wrap_error(error_cls, message, exc, ...)`` helper. It:
    1. Instantiates the requested typed error.
    2. Records ``details={"original": type(exc).__name__}``.
    3. Attaches the original via ``__cause__`` (full traceback preserved).
- **Logging is centralised** inside ``wrap_error`` for consistency.
- **Direct imports only**: Consumers import from the module where the class
  is defined.

See Also
--------
ai_api.core.xai.errors_xai : xAI-specific extensions (auth, rate limits,
    gRPC, thinking mode, multimodal, cache, batch, structured output).
ai_api.core.ollama.errors_ollama : Ollama-specific extensions (GPU/memory
    warnings, local server connection issues).
"""

from __future__ import annotations

import logging
from typing import Any, Type, TypeVar

__all__ = [
    # Root
    "APIError",
    # Logger
    "LoggerError",
    "LoggerInitializationError",
    # Persistence (shared across backends)
    "PersistenceError",
    "PersistenceSettingsError",
    "PersistencePoolError",
    "DatabasePersistenceError",
    "FilePersistenceError",
    "OutputPersistenceError",
    # Database / Postgres
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    # Generic client errors
    "ClientError",
    # Single generic wrapper (the only public way to create + log errors)
    "wrap_error",
]


# ----------------------------------------------------------------------
# 1. Root of the hierarchy
# ----------------------------------------------------------------------
class APIError(Exception):
    """Root of the entire ai_api error hierarchy.

    All other exceptions inherit from this class (directly or indirectly).
    Every instance carries a human-readable ``message`` and an optional
    ``details`` dictionary suitable for structured logging, metrics, or
    API error responses.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    details : dict[str, Any] or None, optional
        Structured context (e.g. ``{"model": "grok-4", "request_id": "..."}``).
        Must never contain secrets. Defaults to empty dict.

    Attributes
    ----------
    message : str
    details : dict[str, Any]
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details})"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation (useful for API responses)."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# ----------------------------------------------------------------------
# 2. Logger errors
# ----------------------------------------------------------------------
class LoggerError(APIError):
    """Errors originating from the logging subsystem (structlog, handlers, config)."""

    pass


class LoggerInitializationError(LoggerError):
    """Logger could not be initialised (missing handlers, invalid config, etc.)."""

    pass


# ----------------------------------------------------------------------
# 3. Persistence errors (shared across file, DB, stdout, settings)
# ----------------------------------------------------------------------
class PersistenceError(APIError):
    """Base exception for all persistence-related failures.

    Covers PostgreSQL, filesystem JSON output, stdout, connection pools,
    and settings resolution. Subclasses exist for more specific diagnosis.
    """

    pass


class PersistenceSettingsError(PersistenceError):
    """Settings resolution or validation failed (bad ``db_url``, missing ``json_dir``, etc.)."""

    pass


class PersistencePoolError(PersistenceError):
    """Failed to acquire or maintain a connection pool to Postgres."""

    pass


class DatabasePersistenceError(PersistenceError):
    """Raised when a PostgreSQL operation fails inside the persistence abstraction
    (connection, insert, query, …). Use ``DatabaseError`` for low-level SQL errors.
    """

    pass


class FilePersistenceError(PersistenceError):
    """Raised when a filesystem operation fails (JSON write, directory creation, …)."""

    pass


class OutputPersistenceError(PersistenceError):
    """Raised when an output operation fails (stdout write, flush, …)."""

    pass


# ----------------------------------------------------------------------
# 4. Database / Postgres errors (low-level SQL)
# ----------------------------------------------------------------------
class DatabaseError(APIError):
    """Errors from PostgreSQL / asyncpg operations.

    Distinct from ``PersistenceError`` so that callers can catch "any DB problem"
    versus "any persistence problem".
    """

    pass


class DatabaseConnectionError(DatabaseError):
    """Failed to connect to or acquire a connection from Postgres."""

    pass


class DatabaseQueryError(DatabaseError):
    """SQL execution error (syntax, constraint violation, timeout, etc.)."""

    pass


# ----------------------------------------------------------------------
# 5. Generic client errors
# ----------------------------------------------------------------------
class ClientError(APIError):
    """Internal errors within any ai_api client (validation, state machine, usage).

    Provider-specific client errors should inherit from both their provider root
    *and* this class (multiple inheritance) so that ``except ClientError`` catches
    all internal client failures regardless of provider.
    """

    pass


# ----------------------------------------------------------------------
# 6. Generic wrapper (the single place logging + wrapping happens)
# ----------------------------------------------------------------------
_APIErrorT = TypeVar("_APIErrorT", bound=APIError)


def wrap_error(
    error_cls: Type[_APIErrorT],
    message: str,
    exc: Exception | None = None,
    *,
    details: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    level: int = logging.ERROR,
) -> _APIErrorT:
    """Wrap any exception into the requested ``APIError`` subclass and log it.

    This is the **only** public factory for creating and logging errors in the
    entire package. All previous thin convenience wrappers have been removed
    in favour of this single, fully generic, statically typed helper.

    It guarantees:

    - Structured ``details`` (including original exception type)
    - Exception chaining via ``__cause__``
    - Centralised logging (easy to change format, add correlation IDs, switch
      to structlog, etc.)

    The ``TypeVar`` bound ensures static type checkers correctly infer the
    return type from the ``error_cls`` argument.

    Parameters
    ----------
    error_cls : Type[_APIErrorT]
        The concrete error class to instantiate (e.g. ``PersistenceError``,
        ``DatabaseQueryError``, ``XAIClientError``, ``OllamaGPUError``).
    message : str
        Human-readable summary of what failed.
    exc : Exception or None, optional
        The original low-level exception (if any). Will be attached as ``__cause__``.
    details : dict, optional
        Extra context to merge into the error's ``details``.
    logger : logging.Logger, optional
        Logger to use. If ``None``, a module-level logger is created.
    level : int, default logging.ERROR
        Log level. Use ``logging.WARNING`` for recoverable cases.

    Returns
    -------
    _APIErrorT
        The wrapped (and already logged) exception instance — statically typed
        as the exact class you requested.

    Examples
    --------
    >>> from ai_api.core.xai.errors_xai import XAIClientError
    >>> try:
    ...     await client.chat(...)
    ... except Exception as e:
    ...     err = wrap_error(
    ...         XAIClientError,
    ...         "Failed to create chat session with xAI",
    ...         e,
    ...         details={"model": "grok-4", "request_id": rid},
    ...         level=logging.WARNING,
    ...     )
    ...     raise err from e
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    details = details or {}
    if exc is not None:
        details.setdefault("original", type(exc).__name__)
        details.setdefault("original_error", str(exc))

    wrapped = error_cls(message, details=details)
    if exc is not None:
        wrapped.__cause__ = exc

    logger.log(
        level,
        message,
        extra={"error_type": error_cls.__name__, "details": details},
    )
    return wrapped
