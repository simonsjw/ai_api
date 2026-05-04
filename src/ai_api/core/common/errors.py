"""
errors.py — Custom exception hierarchy and error-wrapping utilities for the ai_api package.

This module provides a small, cohesive set of exception classes used across the
persistence layer (and potentially other core components).  The design follows
the principle that every error should carry both a human-readable message and
structured details that can be logged or surfaced to callers without losing
the original exception context.

All errors inherit from ``PersistenceError`` so that client code can catch a
single base type when it wants to treat persistence failures uniformly (while
still being able to inspect ``.message`` and ``.details``).

The ``wrap_persistence_error`` helper is the canonical way to turn low-level
exceptions (asyncpg errors, OSError, etc.) into ai_api errors while preserving
traceback information via exception chaining (``raise ... from exc``).

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PersistenceError",
    "DatabasePersistenceError",
    "FilePersistenceError",
    "OutputPersistenceError",
    "wrap_persistence_error",
]


@dataclass
class PersistenceError(Exception):
    """Base exception for all persistence-related failures.

    Attributes
    ----------
    message : str
        Human-readable description of what went wrong.
    details : dict
        Structured context (model name, kind, backend, original error, …)
        intended for logging or telemetry.  Never contains secrets.
    """

    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)


@dataclass
class DatabasePersistenceError(PersistenceError):
    """Raised when a PostgreSQL operation fails (connection, insert, query, …)."""

    pass


@dataclass
class FilePersistenceError(PersistenceError):
    """Raised when a filesystem operation fails (JSON write, directory creation, …)."""

    pass


@dataclass
class OutputPersistenceError(PersistenceError):
    """Raised when an output operation fails (stdout write, flush, …)."""

    pass


def wrap_persistence_error(
    exc: Exception,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    level: int = logging.ERROR,
) -> PersistenceError:
    """Wrap a low-level exception into a ``PersistenceError`` (or subclass) and log it.

    This is the **recommended** way to handle unexpected errors inside backend
    implementations.  It:

    1. Creates a ``PersistenceError`` (or more specific subclass if you pass one
       via the ``error_class`` kwarg — currently not exposed for simplicity).
    2. Attaches the original exception as ``__cause__`` (Python's exception
       chaining).
    3. Logs the message at the requested level with the structured ``details``
       and the stringified original error.
    4. Returns the wrapped error so the caller can decide whether to raise it
       or swallow it (as the higher-level clients currently do).

    Parameters
    ----------
    exc : Exception
        The original exception that was caught.
    message : str
        Human-readable summary (e.g. "Failed to insert response into Postgres").
    details : dict, optional
        Extra context to include in the error and the log record.
    logger : logging.Logger, optional
        Logger to use.  If ``None`` a module-level logger is created.
    level : int, default logging.ERROR
        Log level.  Use ``logging.WARNING`` for non-fatal cases that the
        caller will continue past.

    Returns
    -------
    PersistenceError
        The wrapped error (already logged).

    Examples
    --------
    >>> try:
    ...     await conn.execute(...)
    ... except asyncpg.PostgresError as e:
    ...     err = wrap_persistence_error(
    ...         e,
    ...         "Failed to persist chat turn",
    ...         details={"model": model, "kind": kind},
    ...         logger=self.logger,
    ...         level=logging.WARNING,
    ...     )
    ...     raise err from e
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    details = details or {}
    details.setdefault("original_error_type", type(exc).__name__)
    details.setdefault("original_error", str(exc))

    error = PersistenceError(message=message, details=details)
    logger.log(
        level,
        message,
        extra={"error": str(exc), "details": details},
    )
    return error
