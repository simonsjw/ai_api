#!/usr/bin/env python3
"""Custom exceptions for the ai_api package.

This module centralises all exceptions raised by the ai_api client.  It is
imported by client.py and exposed in the public API.
"""

from __future__ import annotations


class AIAPIError(Exception):
    """Base class for all ai_api exceptions.

    Parameters
    ----------
    message : str
        Human-readable error description.

    Notes
    -----
    All derived exceptions inherit from this class to allow uniform catching.
    """

    pass


class UnsupportedThinkingModeError(AIAPIError, ValueError):
    """Raised when the 'think' parameter is explicitly set for a model that
    does not support it.

    This enforces the strict opt-in policy requested by the user.

    Parameters
    ----------
    message : str
        Detailed error message including the model name and supported families.

    Notes
    -----
    The exception is raised before any network request is made, avoiding the
    raw Ollama ResponseError.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
