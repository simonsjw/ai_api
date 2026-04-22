"""High-level asynchronous client for the xAI API using the official xAI SDK."""

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
from .client_factory import get_llm_client, register_provider
from .common.persistence import PersistenceManager
from .xai.chat_batch_xai import create_batch_chat
from .xai.chat_stream_xai import generate_stream_and_persist
from .xai.chat_turn_xai import create_turn_chat_session

ChatMode = Literal["turn", "stream", "batch"]


class BaseXAIClient:
    """Shared base class containing HTTP client lifecycle logic."""

    def __init__(
        self,
        logger: logging.Logger,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        timeout: Optional[int] = 120,
        persistence_manager: "PersistenceManager" | None = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.logger = logger
        self.persistence_manager = persistence_manager
        self._http_client = None
        self._sdk_client: XAIAsyncClient | None = None

    async def _get_sdk_client(self) -> XAIAsyncClient:
        if self._sdk_client is None:
            self._sdk_client = XAIAsyncClient(api_key=self.api_key)
        return self._sdk_client

    async def aclose(self) -> None:
        pass


class TurnXAIClient(BaseXAIClient):
    async def create_chat(
        self,
        messages: list[dict] | None = None,
        model: str = "grok-4",
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_mode: SaveMode = "none",
        **kwargs: Any,
    ) -> Any:
        """Create a stateful turn-based chat (fully supports multimodal).

        Multimodal example:
            messages = [{"role": "user", "content": [
                {"type": "input_text", "text": "Describe this"},
                {"type": "input_image", "image_url": "https://..."},
                {"type": "input_file",  "file_url": "/path/to/file.pdf"}
            ]}]
        Media files are automatically saved when persistence_manager.media_root is set.
        """
        if self._sdk_client is None:
            self._sdk_client = XAIAsyncClient(api_key=self.api_key)
        return await create_turn_chat_session(
            self,
            self._sdk_client,
            messages or [],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            **kwargs,
        )


class StreamXAIClient(BaseXAIClient):
    """Streaming XAI client built directly on the official xAI SDK (`xai_sdk.AsyncClient`).

    Uses lazy initialisation of the SDK client and delegates real-time token
    streaming (combined with optional persistence) to ``generate_stream_and_persist``.
    """

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
        """Create a streaming chat completion (fully supports multimodal).

        Same media-attachment format as turn mode. Media files are saved at completion.
        """
        sdk_client = await self._get_sdk_client()

        xai_input = xAIInput.from_list(messages)
        request = xAIRequest(
            input=xai_input,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            **kwargs,
        )

        chat = sdk_client.chat.create(
            model=request.model,
            **request.to_sdk_chat_kwargs(),
        )

        async for chunk in generate_stream_and_persist(
            self.logger,
            self.persistence_manager,
            chat,
            request,
            save_mode=save_mode,
        ):
            yield chunk


class BatchXAIClient(BaseXAIClient):
    async def create_chat(
        self,
        messages: list[list[dict]],
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


def XAIClient(
    logger: logging.Logger,
    api_key: str,
    *,
    mode: ChatMode = "turn",
    base_url: str = "https://api.x.ai/v1",
    timeout: Optional[int] = 120,
    persistence_manager: "PersistenceManager" | None = None,
    **kwargs: Any,
) -> BaseXAIClient:
    """Factory – delegates to the central registry (see client_factory.py)."""
    return get_llm_client(
        "xai",
        logger=logger,
        mode=mode,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        persistence_manager=persistence_manager,
        **kwargs,
    )


register_provider("xai", TurnXAIClient, StreamXAIClient, BatchXAIClient)
