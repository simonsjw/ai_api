"""
xAI-Specific Error Hierarchy (Full Backward Compatibility)

This module defines the complete xAI error taxonomy while preserving every
public name from earlier versions of the library. It delegates generic
behaviour to ``core/common/errors.py`` and adds xAI-specific subclasses
for gRPC, rate limiting, thinking mode, multimodal, caching, batch failures,
and structured output parsing errors.

Design Goals
------------
- Consistent with ``ollama/errors_ollama.py`` and the shared base classes.
- No changes required for the new Git-style branching features (errors are
  orthogonal to conversation metadata).

New in 2026
-----------
- ``xAIStructuredOutputError``: Raised when ``response_model`` validation fails
  after a successful xAI API response. This replaces the previous "log and
  continue with raw text" behaviour for stricter, typed error handling.

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
    "xAIStructuredOutputError",                                                           # NEW
    # Wrappers (kept exactly as before)
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


class xAIStructuredOutputError(xAIClientError):
    """Pydantic validation of the model response against response_model failed.

    **Default behaviour**: The library logs a rich WARNING and returns the
    response with raw ``.text`` (``.parsed`` is left unset). This follows the
    "best-effort / continue on non-fatal issues" contract used for persistence
    errors and matches common LLM client behaviour (OpenAI, Anthropic, etc.).

    The exception class is provided so that:
    - Advanced callers can `except xAIStructuredOutputError` if they care.
    - A future `strict_structured_output=True` flag can be added without
      introducing a new exception type.

    The original Pydantic ``ValidationError`` is preserved as ``__cause__``.
    """

    pass


# Wrapper aliase (kept exactly as before)


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
