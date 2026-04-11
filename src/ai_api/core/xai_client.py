import logging
from typing import Any, Literal, Optional, Type

import httpx

from .xai.chat_batch_xai import create_batch_chat
from .xai.chat_multim_xai import create_multim_chat
from .xai.chat_stream_xai import create_stream_chat
from .xai.chat_turn_xai import create_turn_chat
from .xai.persistence_xai import xAIPersistenceManager

ChatMode = Literal["turn", "stream", "batch", "multim"]


class BaseXAIClient:
    """Shared base class containing HTTP client lifecycle logic."""

    def __init__(
        self,
        logger: logging.Logger,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        timeout: Optional[int] = 120,
        persistence_manager: "xAIPersistenceManager" | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the asynchronous xAI client."""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.logger = logger
        self.persistence_manager = persistence_manager
        self._http_client = httpx.AsyncClient(timeout=timeout, **kwargs)

    async def aclose(self) -> None:
        """Explicitly close the underlying HTTP client."""
        await self._http_client.aclose()

    async def __aenter__(self):
        """Support async context manager usage."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the HTTP client on context exit."""
        await self.aclose()

        # Specialised client classes (unchanged)


class TurnXAIClient(BaseXAIClient):
    async def create_chat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        return await create_turn_chat(
            self,
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


class StreamXAIClient(BaseXAIClient):
    async def create_chat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        return await create_stream_chat(
            self,
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


class BatchXAIClient(BaseXAIClient):
    async def create_chat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        return await create_batch_chat(
            self,
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


class MultimXAIClient(BaseXAIClient):
    async def create_chat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        return await create_multim_chat(
            self,
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


def XAIClient(
    logger: logging.Logger,
    api_key: str,
    *,
    mode: ChatMode = "turn",
    base_url: str = "https://api.x.ai/v1",
    timeout: Optional[int] = 120,
    persistence_manager: "xAIPersistenceManager" | None = None,
    **kwargs: Any,
) -> BaseXAIClient:
    """Factory function that creates and returns a specialised xAI client
    for the requested interaction mode.

    This is the primary public entry point to the xAI API library.
    It returns a client instance with a consistent `create_chat(...)`
    interface regardless of the underlying chat modality.

    The `mode` parameter selects the appropriate specialised implementation
    while keeping the public API clean and uniform.

    Args:
        api_key: xAI API key used for authentication.
        mode: Determines the chat interaction type and behaviour of `create_chat`.
            Must be one of: "turn", "stream", "batch", or "multim".
            Defaults to "turn" (standard non-streaming chat).
        base_url: Base URL for the xAI API service.
        timeout: Default HTTP request timeout in seconds.
        **kwargs: Additional arguments passed directly to `httpx.AsyncClient`.

    Returns:
        An instance of the appropriate client subclass (TurnXAIClient,
        StreamXAIClient, etc.) which supports async context management
        and provides `create_chat`.

    Example:
        >>> client = XAIClient(api_key="xai-...", mode="stream")
        >>> async for chunk in client.create_chat(messages=..., model="grok-3"):
        ...     print(chunk)
    """
    client_map: dict[ChatMode, Type[BaseXAIClient]] = {
        "turn": TurnXAIClient,
        "stream": StreamXAIClient,
        "batch": BatchXAIClient,
        "multim": MultimXAIClient,
    }

    ClientClass = client_map.get(mode)
    if ClientClass is None:
        raise ValueError(
            f"Unsupported mode '{mode}'. Must be one of: {list(client_map.keys())}"
        )

    return ClientClass(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        persistence_manager=persistence_manager,
        **kwargs,
    )
