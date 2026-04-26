"""
Central registry-based factory for LLM clients (Ollama, xAI, and future providers).

This module is the **single source of truth** for obtaining clients. It eliminates
the duplicated mode-dispatch logic that previously lived in both
``ollama_client.py`` and ``xai_client.py``.

Architectural role
------------------
- **Registry pattern**: Providers self-register at import time by calling
  ``register_provider(...)`` at the bottom of their ``*_client.py`` files.
- **Lazy auto-registration**: ``get_llm_client("ollama")`` or ``get_llm_client("xai")``
  will automatically import the provider module if it hasn't been registered yet.
- **Unified API**: Users can choose the classic style (``OllamaClient(...)`` /
  ``XAIClient(...)``) **or** the generic style (``get_llm_client("ollama", mode=...)``).
- Adding a new provider (e.g. Groq, Anthropic, vLLM) requires only three lines:
  define the three mode clients, then call ``register_provider(...)``.

How it uses the rest of the core/ structure
-------------------------------------------
- Imports from ``common/persistence.py`` (via the provider clients).
- Delegates to provider-specific modules in ``ollama/`` and ``xai/``
  (``chat_turn_*.py``, ``chat_stream_*.py``, ``chat_batch_xai.py``, embeddings, etc.).
- The returned clients satisfy ``LLMProviderAdapter`` (from ``base_provider.py``).
- All clients share the same ``PersistenceManager``, ``SaveMode``, error
  wrappers, and ``response_model`` structured-output pattern.

Example usage (recommended unified style)
-----------------------------------------
.. code-block:: python

    from ai_api.core.client_factory import get_llm_client
    import logging

    logger = logging.getLogger(__name__)
    pm = PersistenceManager(logger=logger, db_url=...)

    # Ollama streaming
    ollama = get_llm_client(
        "ollama",
        logger=logger,
        mode="stream",
        host="http://localhost:11434",
        persistence_manager=pm,
    )
    async for chunk in ollama.create_chat(messages=..., model="llama3.2"):
        ...

    # xAI batch with per-request structured output
    xai = get_llm_client("xai", logger=logger, mode="batch", api_key=...)
    results = await xai.create_chat(
        messages_list=[...], model="grok-4", response_model=[PersonModel, SummaryModel, ...]
    )

See Also
--------
ai_api.core.base_provider.LLMProviderAdapter
ai_api.core.ollama_client.py, ai_api.core.xai_client.py
    The concrete clients that get registered.
ai_api.core.common
    Shared persistence, errors, and structured-output helpers used by all.
"""

import logging
from typing import Any, Literal, Type

__all__: list[str] = [
    "ChatMode",
    "PROVIDER_REGISTRY",
    "register_provider",
    "get_llm_client",
]


ChatMode = Literal["turn", "stream", "batch"]

# provider_name -> {mode: ClientClass}
PROVIDER_REGISTRY: dict[str, dict[ChatMode, Type]] = {}


def register_provider(
    name: str,
    turn_cls: Type,
    stream_cls: Type,
    batch_cls: Type,
) -> None:
    """Register the three client classes for a new LLM provider.

    Call this once (usually at the bottom of the provider's ``*_client.py`` file)
    after the classes are defined.

    Parameters
    ----------
    name : str
        Provider identifier (e.g. "ollama", "xai", "groq").
    turn_cls, stream_cls, batch_cls : Type
        The three mode-specific client classes.
    """
    if name in PROVIDER_REGISTRY:
        raise ValueError(f"Provider '{name}' is already registered.")
    PROVIDER_REGISTRY[name] = {
        "turn": turn_cls,
        "stream": stream_cls,
        "batch": batch_cls,
    }


def get_llm_client(
    provider: str,
    logger: logging.Logger,
    mode: ChatMode = "turn",
    **kwargs: Any,
) -> Any:
    """Unified factory — returns the appropriate *Client instance for any provider.

    Parameters
    ----------
    provider : str
        "ollama", "xai", or any name previously passed to ``register_provider``.
    logger : logging.Logger
        Your structured logger instance.
    mode : {"turn", "stream", "batch"}, optional
        Desired interaction mode (default "turn").
    **kwargs
        Provider-specific constructor arguments
        (host, api_key, base_url, timeout, persistence_manager, etc.).

    Returns
    -------
    Turn*Client | Stream*Client | Batch*Client
        The exact type depends on provider + mode. All satisfy
        ``LLMProviderAdapter``.

    Raises
    ------
    ValueError
        Unknown provider or unsupported mode for that provider.
    """
    if provider not in PROVIDER_REGISTRY:
        # Lazy auto-register for built-in providers
        if provider == "ollama":
            from . import ollama_client                                                   # noqa: F401 - triggers register_provider
        elif provider == "xai":
            from . import xai_client                                                      # noqa: F401 - triggers register_provider
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Currently registered: {list(PROVIDER_REGISTRY.keys())}. "
                "Did you forget to call register_provider(...) or import the provider module?"
            )

    mode_map = PROVIDER_REGISTRY[provider]
    ClientClass: Type = mode_map.get(mode)                                                # type: ignore[assignment]
    if ClientClass is None:
        raise ValueError(
            f"Mode '{mode}' is not supported for provider '{provider}'. "
            f"Supported modes: {list(mode_map.keys())}"
        )

    # Instantiate the chosen class. All provider classes accept the same common kwargs
    # (logger, persistence_manager, timeout, ...) plus their own required ones.
    return ClientClass(logger=logger, **kwargs)                                           # type: ignore[call-arg]
