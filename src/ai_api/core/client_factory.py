"""Central registry-based factory for LLM clients (Ollama, xAI, and future providers).

This eliminates the duplicated mode-dispatch logic that previously lived
in both ollama_client.py and xai_client.py.

Key benefits:
- Single place to add new providers (register_provider + 3 client classes).
- Backward compatible: existing OllamaClient(...) and XAIClient(...) still work.
- Unified get_llm_client("provider", mode=..., **kwargs) for generic code.

Example - existing style (still supported):
    client = OllamaClient(logger=logger, mode="stream", host=..., persistence_manager=pm)

Example - new unified style:
    from ai_api.core.client_factory import get_llm_client, register_provider
    client = get_llm_client(
        "ollama",
        logger=logger,
        mode="stream",
        host="http://localhost:11434",
        persistence_manager=pm,
    )

    # Later, for a new provider (e.g. Groq)
    register_provider("groq", GroqTurnClient, GroqStreamClient, GroqBatchClient)
    client = get_llm_client("groq", logger=logger, mode="turn", api_key=...)
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Type

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

    Call this once (usually at the bottom of the provider's *_client.py file)
    after the classes are defined.
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

    Args:
        provider: "ollama", "xai", or any name previously passed to register_provider.
        logger: Your structured logger instance.
        mode: "turn" | "stream" | "batch"
        **kwargs: Provider-specific constructor arguments
                  (host, api_key, base_url, timeout, persistence_manager, etc.)

    Returns:
        Turn*Client | Stream*Client | Batch*Client instance (exact type depends on provider+mode).

    Raises:
        ValueError: Unknown provider or unsupported mode for that provider.
    """
    if provider not in PROVIDER_REGISTRY:
        # Lazy auto-register for built-in providers (so get_llm_client("ollama") works
        # even if the user never imported ollama_client directly)
        if provider == "ollama":
            from . import ollama_client  # noqa: F401 - triggers register_provider at bottom
        elif provider == "xai":
            from . import xai_client  # noqa: F401 - triggers register_provider at bottom
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Currently registered: {list(PROVIDER_REGISTRY.keys())}. "
                "Did you forget to call register_provider(...) or import the provider module?"
            )

    mode_map = PROVIDER_REGISTRY[provider]
    ClientClass: Type = mode_map.get(mode)  # type: ignore[assignment]
    if ClientClass is None:
        raise ValueError(
            f"Mode '{mode}' is not supported for provider '{provider}'. "
            f"Supported modes: {list(mode_map.keys())}"
        )

    # Instantiate the chosen class. All provider classes accept the same common kwargs
    # (logger, persistence_manager, timeout, ...) plus their own required ones.
    return ClientClass(logger=logger, **kwargs)  # type: ignore[call-arg]
