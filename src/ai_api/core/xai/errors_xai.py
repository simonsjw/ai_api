"""
xAI-specific error hierarchy.

This module defines exceptions specific to the xAI (Grok) remote managed
service. All exceptions inherit from ``xAIError`` (which inherits from
``ai_api.core.common.errors.AIAPIError``).

High-level design
-----------------
- **Root**: ``xAIError`` – catches everything xAI-related.
- **API / Transport layer**: ``xAIAPIError`` and subclasses. Includes native
  support for gRPC (when the ``grpc`` package is installed) because parts of
  the xAI SDK and internal transport use gRPC. Conditional classes allow
  catching as either ``xAIAPI*Error`` or ``grpc.RpcError``.
- **Client layer**: ``xAIClientError`` and a rich set of domain-specific
  subclasses that reflect remote-service concerns absent from local providers:
  authentication, rate limiting, thinking mode, multimodal inputs, prompt
  caching, and native batch processing.

The 'wrap error' approach (identical contract to common and Ollama)
-------------------------------------------------------------------
Raw exceptions from the xAI SDK, ``httpx``, or ``grpc`` are converted by the
provided wrappers:

.. code-block:: python

    from ai_api.core.xai.errors import wrap_xai_api_error

    try:
        response = await sdk_client.chat.create(...)
    except Exception as exc:
        raise wrap_xai_api_error(exc, "xAI chat completion failed") from exc

Each wrapper:
1. Instantiates the most specific typed error with a clear ``message``.
2. Records the original exception class name in ``details["original"]``.
3. Attaches ``__cause__`` so ``traceback`` / logging libraries see the full
   causal chain.

Direct import pattern (no pass-through convenience importing)
-------------------------------------------------------------
**Always import from the defining module.** There are no legacy aliases or
re-exports of common errors under ``xAI*`` names (those were removed in this
refactor to enforce clarity).

.. code-block:: python

    from ai_api.core.common.errors import AIAPIError, AIClientError, wrap_client_error
    from ai_api.core.xai.errors import (
        xAIError,
        xAIAPIError,
        xAIAPIRateLimitError,
        xAIClientMultimodalError,
        wrap_xai_api_error,
    )

This design makes static type checkers, IDEs, and code reviewers instantly
aware of where each exception originates.

Comparison with Ollama errors (highlighting additional xAI features)
--------------------------------------------------------------------
- **xAI (remote)**: richer taxonomy because it is a managed cloud service.
  Additional error domains (not present in Ollama):
    - Authentication & credential errors
    - Rate-limit / quota errors
    - gRPC transport errors (conditional)
    - Thinking-mode compatibility errors
    - Multimodal (vision) input errors
    - Prompt-cache / context-cache errors
    - Native batch-processing errors
- **Ollama (local)**: simpler HTTP surface + one GPU/memory warning that has
  no equivalent here (remote providers hide infrastructure details).

Both modules follow identical documentation style, wrap contract, and import
discipline so that adding a third provider later is mechanical.

See Also
--------
ai_api.core.common.errors : Canonical base classes and the four generic
    ``wrap_*_error`` helpers.
ai_api.core.ollama.errors : Local-provider counterpart (includes
    ``OllamaGPUMemoryWarning``).
"""

from __future__ import annotations

try:
    import grpc
except ImportError:
    grpc = None                                                                           # type: ignore[assignment]

# Direct import of shared bases only – no re-export of common errors
from ..common.errors import (
    AIAPIError,
    AIClientError,
)

__all__: list[str] = [
    "xAIError",
    # API / transport layer
    "xAIAPIError",
    "xAIAPIConnectionError",
    "xAIAPIAuthenticationError",
    "xAIAPIInvalidRequestError",
    "xAIAPIRateLimitError",
    # Client layer (domain-specific)
    "xAIClientError",
    "UnsupportedThinkingModeError",
    "xAIClientBatchError",
    "xAIClientMultimodalError",
    "xAIClientCacheError",
    # Wrap helpers
    "wrap_xai_error",
    "wrap_xai_api_error",
]


# ----------------------------------------------------------------------
# 1. Root of all xAI errors
# ----------------------------------------------------------------------
class xAIError(AIAPIError):
    """Root of all xAI-related exceptions.

    Every exception raised by ``XAIClient``, the chat/embedding/batch helpers,
    or model-info methods inherits from this class.
    """

    pass


# ----------------------------------------------------------------------
# 2. API / Transport errors (remote xAI service)
# ----------------------------------------------------------------------
class xAIAPIError(xAIError):
    """Errors returned by the official xAI API (HTTP or gRPC transport)."""

    pass


if grpc is not None:
    # gRPC-aware subclasses (when grpc is installed) – allows dual catching
    class xAIAPIConnectionError(xAIAPIError, grpc.RpcError):
        """gRPC connection or transport-level failure to the xAI service."""

        pass

    class xAIAPIAuthenticationError(xAIAPIError, grpc.RpcError):
        """gRPC authentication failure (invalid/expired API key, insufficient
        permissions, etc.).
        """

        pass
else:

    class xAIAPIConnectionError(xAIAPIError):
        """Failed to connect to the xAI API endpoint (network, DNS, TLS, etc.)."""

        pass

    class xAIAPIAuthenticationError(xAIAPIError):
        """Authentication failed (bad API key, revoked token, account issues)."""

        pass


class xAIAPIInvalidRequestError(xAIAPIError):
    """xAI rejected the request (unknown model, invalid parameters, prompt too
    long, unsupported feature combination, etc.).
    """

    pass


class xAIAPIRateLimitError(xAIAPIError):
    """Rate limit, quota, or abuse-protection threshold exceeded on the xAI
    service. The ``details`` dict usually contains ``retry_after`` when
    provided by the API.
    """

    pass


# ----------------------------------------------------------------------
# 3. Client-internal errors (XAIClient logic) – domain-specific extensions
# ----------------------------------------------------------------------
class xAIClientError(xAIError, AIClientError):
    """Internal errors within ``XAIClient`` or its mode-specific subclasses
    (Turn, Stream, Batch, Embed).

    These are not transport errors; they represent logic, validation, or
    state-machine problems inside the client library.
    """

    pass


class UnsupportedThinkingModeError(xAIClientError):
    """Requested thinking mode (e.g. ``thinking_mode="high"``) is not supported
    by the chosen model or the current version of the xAI SDK.
    """

    pass


class xAIClientBatchError(xAIClientError):
    """Error specific to native batch processing.

    Common triggers: length mismatch between ``messages_list`` and
    ``response_model`` list, or an individual item failing while others succeed.
    """

    pass


class xAIClientMultimodalError(xAIClientError):
    """Error during multimodal (image, video, or document) input handling.

    Covers invalid image formats, oversized payloads, unsupported MIME types,
    or vision-model capability mismatches.
    """

    pass


class xAIClientCacheError(xAIClientError):
    """Error related to prompt caching or context caching features.

    Includes cache-miss handling failures, cache-key collisions, or
    server-side cache quota issues.
    """

    pass


# ----------------------------------------------------------------------
# Convenience wrapper functions (xAI-specific)
# ----------------------------------------------------------------------
def wrap_xai_error(exc: Exception, message: str) -> xAIError:
    """Generic wrapper for any xAI-related exception.

    Use when the precise subcategory is not known at the catch site.

    Parameters
    ----------
    exc : Exception
        The original exception (xAI SDK, httpx, grpc, etc.).
    message : str
        Human-readable description of the operation that failed.

    Returns
    -------
    xAIError
        Properly typed exception ready for logging or re-raising.
    """
    wrapped = xAIError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped


def wrap_xai_api_error(exc: Exception, message: str) -> xAIAPIError:
    """Wrap errors originating from the xAI SDK, gRPC layer, or HTTP transport.

    This is the most commonly used wrapper inside ``chat_turn_xai.py``,
    ``chat_stream_xai.py``, and ``chat_batch_xai.py``.

    Parameters
    ----------
    exc : Exception
        The original low-level exception.
    message : str
        Contextual message describing what the client was attempting.

    Returns
    -------
    xAIAPIError
    """
    wrapped = xAIAPIError(message, details={"original": type(exc).__name__})
    wrapped.__cause__ = exc
    return wrapped
