import logging
from collections.abc import AsyncIterator
from typing import Any, Literal, Optional, Type

from xai_sdk import AsyncClient as XAIAsyncClient

from ..data_structures.xai_objects import (
    LLMStreamingChunkProtocol,
    SaveMode,
    xAIInput,
    xAIRequest,
)
from .xai.chat_batch_xai import create_batch_chat
from .xai.chat_multim_xai import create_multim_chat
from .xai.chat_stream_xai import create_stream_chat
from .xai.chat_turn_xai import create_turn_chat_session
from .xai.persistence_xai import xAIPersistenceManager
from .xai.stream_xai import generate_stream_and_persist

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
        self._http_client = None                                                          # retained only for non-turn modes
        self._sdk_client: XAIAsyncClient | None = None

    async def _get_sdk_client(self) -> XAIAsyncClient:
        if self._sdk_client is None:
            self._sdk_client = XAIAsyncClient(
                api_key=self.api_key
            )                                                                             # SDK handles base_url, timeouts, retries
        return self._sdk_client

    async def aclose(self) -> None:
        # SDK clients are lightweight; no explicit close required in most cases
        pass


class TurnXAIClient(BaseXAIClient):
    async def create_chat(
        self,
        messages: list[dict] | None = None,
        model: str = "grok-4",
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a stateful turn-based chat using the official xAI SDK."""
        if self._sdk_client is None:
            self._sdk_client = XAIAsyncClient(api_key=self.api_key)
        return await create_turn_chat_session(
            self,
            self._sdk_client,
            messages or [],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


class StreamXAIClient(BaseXAIClient):
    """Streaming client using the official xAI SDK."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sdk_client: XAIAsyncClient | None = None

    async def _get_sdk_client(self) -> XAIAsyncClient:
        """Lazy-initialise the official xAI AsyncClient (shares auth with HTTP client)."""
        if self._sdk_client is None:
            self._sdk_client = XAIAsyncClient(api_key=self.api_key)
        return self._sdk_client

    async def create_chat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_mode: SaveMode = "none",
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamingChunkProtocol]:
        """Create a streaming chat completion using the official xAI SDK.

        Yields individual `LLMStreamingChunkProtocol` objects in real time.
        If `save_mode="postgres"` the request is persisted before streaming
        and the final accumulated response is persisted once at completion.
        """
        sdk_client = await self._get_sdk_client()

        # Build canonical request object (reuses existing models)
        xai_input = xAIInput.from_list(messages)
        request = xAIRequest(
            input=xai_input,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            **kwargs,
        )

        # Create SDK chat object (official pattern)
        chat = sdk_client.chat.create(
            model=request.model,
            **request.to_api_kwargs(),                                                    # reuse your existing helper if it maps correctly
        )

        # Delegate to updated generate_stream_and_persist (now SDK-based)
        async for chunk in generate_stream_and_persist(
            sdk_client, chat, request, save_mode=save_mode
        ):
            yield chunk


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
        logger: logging object
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
        logger=logger,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        persistence_manager=persistence_manager,
        **kwargs,
    )
