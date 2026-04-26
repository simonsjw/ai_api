"""
LLMProviderAdapter — structural Protocol for all LLM clients.

This tiny module defines the contract that every client implementation
must satisfy. Because it uses ``@runtime_checkable`` Protocol with structural
subtyping, client classes do **not** need to inherit from it — they only need
to implement a ``create_chat`` method with a compatible signature.

Role in the architecture
------------------------
- Provides a common type for generic code that wants to accept "any LLM client"
  without caring whether it is Ollama, xAI, or a future provider.
- Used by the factory, by higher-level orchestration code, and by the
  generic helpers in ``common/response_struct.py`` (``generate_structured_json`` etc.).
- The loose typing (``*args, **kwargs`` → ``Any``) is intentional: it accommodates
  the very different signatures of turn / stream / batch modes across providers.

How it relates to the rest of core/
-----------------------------------
- ``client_factory.py`` returns instances that satisfy this protocol.
- ``ollama_client.py`` and ``xai_client.py`` (and their mode-specific subclasses)
  automatically satisfy it because they define ``create_chat``.
- The provider-specific modules in ``ollama/`` and ``xai/`` implement the
  actual logic that the top-level clients delegate to.

See Also
--------
ai_api.core.client_factory
ai_api.core.ollama_client, ai_api.core.xai_client
ai_api.core.common.response_struct
    Generic helpers that accept any object satisfying this protocol.
"""

from typing import Any, Protocol, runtime_checkable

__all__: list[str] = ["LLMProviderAdapter"]


@runtime_checkable
class LLMProviderAdapter(Protocol):
    """Structural protocol for LLM provider clients.

    Any class that defines a ``create_chat`` method with a roughly compatible
    signature will satisfy this protocol automatically (no inheritance required).

    We intentionally use very loose typing here so it works with the
    existing Turn / Stream / Batch implementations without modification.

    Methods
    -------
    create_chat(*args, **kwargs) -> Any
        Core entry point. Signature varies by mode and provider.
    """

    async def create_chat(self, *args: Any, **kwargs: Any) -> Any:
        """Core chat entry point (turn, stream, or batch).

        The actual signature varies by mode and provider, so we accept
        anything and return anything. This is the pragmatic approach
        for a multi-mode, multi-provider library.
        """
        ...
