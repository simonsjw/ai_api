"""
LLMProviderAdapter — final lenient version using Protocol.

IMPORTANT: Do NOT inherit from this class in your client files.
Because it is a Protocol, Python uses structural subtyping.
As long as your classes have a `create_chat` method, they automatically
satisfy the protocol (no inheritance needed).

This completely avoids "bad-override" errors from strict type checkers
like Pyrefly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..data_structures.base_objects import SaveMode


@runtime_checkable
class LLMProviderAdapter(Protocol):
    """Structural protocol for LLM provider clients.

    Any class that defines a `create_chat` method with a roughly compatible
    signature will satisfy this protocol automatically.

    We intentionally use very loose typing here so it works with the
    existing Turn / Stream / Batch implementations without modification.
    """

    async def create_chat(self, *args: Any, **kwargs: Any) -> Any:
        """Core chat entry point (turn, stream, or batch).

        The actual signature varies by mode and provider, so we accept
        anything and return anything. This is the pragmatic approach
        for a multi-mode, multi-provider library.
        """
        ...
