"""
Ollama-Specific Error Hierarchy

This module defines Ollama-specific error classes that inherit from the shared
base classes in ``ai_api.core.common.errors``.

All Ollama errors ultimately inherit from ``APIError``, so a single
``except APIError`` catches everything. Provider-specific catching remains
possible via ``except OllamaError`` or ``except OllamaClientError``.

**Usage of the generic wrapper**

All error creation and logging should use the single generic factory from the
common module:

    from ai_api.core.common.errors import wrap_error
    from .errors_ollama import OllamaClientError, OllamaGPUError

    err = wrap_error(
        OllamaClientError,
        "Failed to connect to local Ollama server",
        exc,
        details={"model": model, "host": ollama_host},
    )
    raise err from exc

Provider-specific concerns covered here
---------------------------------------
- Local server connection and timeout issues
- GPU / memory exhaustion on the host running Ollama
- Model not found or loading failures
- Quantization / context length problems

See Also
--------
ai_api.core.common.errors : Shared base classes and the generic ``wrap_error``.
ai_api.core.xai.errors_xai : Equivalent module for the xAI provider.
"""

from ..common.errors import APIError, ClientError, PersistenceError


__all__ = [
    "OllamaError",
    "OllamaClientError",
    "OllamaGPUError",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaContextLengthError",
]


class OllamaError(APIError):
    """Base class for all Ollama-specific errors.

    All Ollama errors inherit from this class (and ultimately from ``APIError``).
    """

    pass


class OllamaClientError(ClientError, OllamaError):
    """Internal client errors for the Ollama provider (validation, state machine,
    request building, response parsing, local server communication, etc.).

    Multiple inheritance allows catching with either ``except ClientError``
    (all providers) or ``except OllamaClientError`` (Ollama only).
    """

    pass


class OllamaGPUError(OllamaError):
    """GPU or system memory exhaustion while running a model on the local Ollama
    instance.

    This is typically raised when the requested model is too large for the
    available VRAM / RAM on the host machine.
    """

    pass


class OllamaConnectionError(OllamaError):
    """Failed to connect to or communicate with the local Ollama server
    (connection refused, timeout, etc.).
    """

    pass


class OllamaModelNotFoundError(OllamaError):
    """The requested model is not available on the local Ollama instance."""

    pass


class OllamaContextLengthError(OllamaError):
    """The prompt exceeds the maximum context length supported by the model
    or the Ollama server configuration.
    """

    pass
