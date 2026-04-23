"""
xai_client.py — updated with response_model support for all modes (including batch).

Changes:
- BatchXAIClient.create_chat now accepts response_model (single or list)
- XAIClient factory accepts and forwards response_model
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Literal, Optional, Type

from pydantic import BaseModel
from xai_sdk import AsyncClient as XAIAsyncClient

from ..data_structures.xai_objects import (
    LLMStreamingChunkProtocol,
    SaveMode,
    xAIInput,
    xAIRequest,
)
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
        response_model: Type[BaseModel] | None = None,                                    # ← NEW
        **kwargs: Any,
    ) -> Any:
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
            response_model=response_model,
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
        save_mode: SaveMode = "none",
        response_model: Type[BaseModel] | None = None,                                    # ← NEW
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamingChunkProtocol]:
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
            response_model=response_model,
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
        save_mode: SaveMode = "none",
        response_model: list[Type[BaseModel]]
        | Type[BaseModel]
        | None = None,                                                                    # ← FLEXIBLE
        **kwargs: Any,
    ) -> Any:
        return await create_batch_chat(
            self,
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            save_mode=save_mode,
            response_model=response_model,
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
    response_model: Any = None,                                                           # ← ACCEPTED AT FACTORY LEVEL
    **kwargs: Any,
) -> BaseXAIClient:
    """Factory function returning a specialised xAI client.

    response_model is forwarded to the chosen client.
    For batch mode it accepts either a single model or a list of models.
    """
    client_map: dict[ChatMode, Type[BaseXAIClient]] = {
        "turn": TurnXAIClient,
        "stream": StreamXAIClient,
        "batch": BatchXAIClient,
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
        response_model=response_model,                                                    # ← FORWARDED
        **kwargs,
    )
