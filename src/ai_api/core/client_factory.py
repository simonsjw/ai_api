"""
Central registry-based factory for LLM clients (Ollama, xAI, and future providers).

This module provides the low-level unified factory for obtaining mode-specific
clients. For most applications — especially those using persistence, branching,
forking, or history editing — you should use the higher-level
``ChatSession`` (from ``common.chat_session``) instead.

Architectural role
------------------
- Registry pattern with lazy auto-registration of built-in providers.
- All returned clients satisfy ``LLMProviderAdapter``.
- Still useful for one-off calls or when you need the raw mode client directly.
- For branched/editable conversations, use ``ChatSession`` (recommended).

How it uses the rest of the core/ structure
-------------------------------------------
- Imports from ``common/persistence.py`` (via the provider clients).
- Delegates to provider-specific modules in ``ollama/`` and ``xai/``
  (``chat_turn_*.py``, ``chat_stream_*.py``, ``chat_batch_xai.py``, embeddings, etc.).
- The returned clients satisfy ``LLMProviderAdapter`` (from ``base_provider.py``).
- All clients share the same ``PersistenceManager``, ``SaveMode``, and error wrappers.

Example usage
-----------------------------------------
.. code-block:: python

    from ai_api.core.client_factory import get_llm_client
    from ai_api.core.common.chat_session import ChatSession
    from ai_api.core.common.persistence import PersistenceManager
    import logging

    logger = logging.getLogger(__name__)
    pm = PersistenceManager(logger=logger, db_url=...)

    # Recommended: ChatSession (handles branching, editing, state automatically)
    client = get_llm_client("ollama", logger=logger, persistence_manager=pm)
    session = ChatSession(client, pm)

    resp, meta = await session.create_or_continue(
        "Explain quantum entanglement", model="llama3.2"
    )

    # Later: edit history (creates new immutable branch)
    await session.edit_history(
        edit_ops=[{"op": "remove_turns", "indices": [0]}], new_branch_name="cleaned"
    )

    # Low-level / one-off use (still supported)
    ollama = get_llm_client("ollama", logger=logger, mode="stream", persistence_manager=pm)
    async for chunk in ollama.create_chat(messages=..., model="llama3.2"):
        ...

    # xAI batch with per-request structured output
    xai = get_llm_client("xai", logger=logger, mode="batch", api_key=...)
    results = await xai.create_chat(...)

See Also
--------
ai_api.core.base_provider.LLMProviderAdapter
ai_api.core.common.chat_session.ChatSession   (recommended for branching)
ai_api.core.ollama_client, ai_api.core.xai_client
"""

import logging
from typing import Any, Literal, Type, TypedDict

__all__: list[str] = [
    "ChatMode",
    "PROVIDER_REGISTRY",
    "register_provider",
    "get_llm_client",
]


ChatMode: ChatMode = Literal["turn", "stream", "batch", "embed"]


class ProviderModeMap(TypedDict, total=False):
    """Inner map for one provider's mode → client class."""

    turn: Type[Any]
    stream: Type[Any]
    batch: Type[Any]
    embed: Type[Any] | None


# provider_name -> {mode: ClientClass}
PROVIDER_REGISTRY: dict[str, ProviderModeMap] = {}


def register_provider(
    name: str,
    turn_cls: Type,
    stream_cls: Type,
    batch_cls: Type,
    embed_cls: Type | None = None,                                                        # optional for providers without embeddings
) -> None:
    """Register the three client classes for a new LLM provider.

    Call this once (usually at the bottom of the provider's ``*_client.py`` file)
    after the classes are defined.

    Parameters
    ----------
    name : str
        Provider identifier (e.g. "ollama", "xai", "groq").
    turn_cls, stream_cls, batch_cls, embed_cls : Type
        The four mode-specific client classes.
    """
    if name in PROVIDER_REGISTRY:
        raise ValueError(f"Provider '{name}' is already registered.")
    PROVIDER_REGISTRY[name] = {
        "turn": turn_cls,
        "stream": stream_cls,
        "batch": batch_cls,
        "embed": embed_cls,
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
    mode : {"turn", "stream", "batch", "embed"}, optional
        Desired interaction mode (default "turn").
    **kwargs
        Provider-specific constructor arguments
        (host, api_key, base_url, timeout, persistence_manager, etc.).

    Returns
    -------
    Turn*Client | Stream*Client | Batch*Client | Embed*Client
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
        # elif provider == "groq":                                                        # <-- add for every new provider
        #     from . import groq_client
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
