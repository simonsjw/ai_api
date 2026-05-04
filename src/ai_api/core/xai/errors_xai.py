"""
xAI-Specific Error Hierarchy

This module defines xAI-specific error classes that inherit from the shared
base classes in ``ai_api.core.common.errors``.

All xAI errors ultimately inherit from ``APIError``, so a single
``except APIError`` catches everything. Provider-specific catching remains
possible via ``except XAIError`` or ``except XAIClientError``.

**Usage of the generic wrapper**

All error creation and logging should use the single generic factory:

    from ai_api.core.common.errors import wrap_error
    from .errors_xai import XAIClientError, XAIRateLimitError

    err = wrap_error(
        XAIClientError,
        "Failed to call xAI chat endpoint",
        exc,
        details={"model": model, "request_id": request_id},
    )
    raise err from exc

This replaces all previous thin ``wrap_xai_*`` helpers.

Provider-specific concerns covered here
---------------------------------------
- Authentication and API key issues
- Rate limiting (with retry-after information)
- gRPC / streaming connection errors
- Thinking mode, multimodal, caching, batch, and structured output failures

See Also
--------
ai_api.core.common.errors : Shared base classes and the generic ``wrap_error``.
ai_api.core.ollama.errors_ollama : Equivalent module for the Ollama provider.
"""

try:
    import grpc
except ImportError:
    grpc = None  # type: ignore[assignment]


from ..common.errors import APIError, ClientError


__all__ = [
    "XAIError",
    "XAIClientError",
    "XAIRateLimitError",
    "XAIAuthError",
    "XAIAPIConnectionError",
    "XAIAPIInvalidRequestError",
    "UnsupportedThinkingModeError",
    "XAIClientBatchError",
    "XAIClientMultimodalError",
    "XAIClientCacheError",
    "XAIStructuredOutputError",
]


class XAIError(APIError):
    """Base class for all xAI-specific errors.

    All xAI errors inherit from this class (and ultimately from ``APIError``).
    """

    pass


class XAIClientError(ClientError, XAIError):
    """Internal client errors for the xAI provider (validation, state machine,
    request building, response parsing, etc.).

    Multiple inheritance allows catching with either ``except ClientError``
    (all providers) or ``except XAIClientError`` (xAI only).
    """

    pass


class XAIRateLimitError(XAIError):
    """xAI rate limit exceeded.

    The ``details`` dict typically contains ``retry_after`` (seconds) when
    available from the API response.
    """

    pass


class XAIAuthError(XAIError):
    """Authentication or API-key related failure with the xAI service."""

    pass


if grpc is not None:

    class XAIAPIConnectionError(XAIError, grpc.RpcError):
        """gRPC connection or streaming error while talking to xAI."""

        pass
else:

    class XAIAPIConnectionError(XAIError):
        """gRPC connection or streaming error while talking to xAI
        (fallback when grpc is not installed).
        """

        pass


class XAIAPIInvalidRequestError(XAIError):
    """The request sent to xAI was rejected as invalid (bad parameters, etc.)."""

    pass


class UnsupportedThinkingModeError(XAIClientError):
    """Requested thinking mode is not supported by the current model or SDK version."""

    pass


class XAIClientBatchError(XAIClientError):
    """Error specific to batch processing (e.g. mismatched response_model list length)."""

    pass


class XAIClientMultimodalError(XAIClientError):
    """Error related to multimodal (image/video) input handling."""

    pass


class XAIClientCacheError(XAIClientError):
    """Error related to prompt caching or context caching features."""

    pass


class XAIStructuredOutputError(XAIClientError):
    """Pydantic validation of the model response against ``response_model`` failed.

    This exception is raised (or can be caught) when strict structured output
    validation is enabled. The original Pydantic ``ValidationError`` is
    preserved as ``__cause__``.
    """

    pass
