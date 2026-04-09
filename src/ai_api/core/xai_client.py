"""
xAIClient – native xAI SDK client with Responses API, infopypg persistence,
logger, multimodal/file support, prompt caching, batch processing, and
comprehensive error handling via xai_error.py.

This module provides the `xAIClient` class, a complete wrapper around the
official xAI AsyncClient. It uses the modern stateful Responses API
(/v1/responses endpoint) via store_messages=True, enabling server-side message
persistence and improved reasoning handling. Prompt caching is optimised
through x-grok-conv-id gRPC metadata (set via conversation_id), which routes
requests to the same backend server for maximum cache-hit rates.

Key functionality categories
----------------------------
- **Single generation** (`generate`): text, multimodal, or reasoning-enabled
  calls using the modern Responses API with optional PostgreSQL persistence.
- **Streaming generation** (`_generate_stream_and_persist`): real-time chunk
  yielding combined with final accumulated response persistence.
- **Batch processing** (`create_batch`, `add_to_batch`,
  `retrieve_and_persist_batch_results`): full lifecycle support with per-item
  request/response rows linked by `batch_id`.
- **Database persistence**: requests are saved before the API call;
  responses (including reasoning traces) are saved after success. All rows
  use composite foreign keys and JSONB metadata for efficient querying.
- **Prompt caching**: automatic via x-grok-conv-id metadata derived from
  conversation_id (recommended to match xAIRequest.prompt_cache_key).

Main class
----------
xAIClient
    The primary public interface. All public methods are asynchronous.

Important instance attributes
-----------------------------
- `api_key` (str): xAI API key (from argument or XAI_API_KEY environment
  variable).
- `logger` (Logger): mandatory structured logger for every operation.
- `conversation_id` (str | None): used as x-grok-conv-id for prompt caching
  and server affinity.
- `_client` (xai_sdk.AsyncClient): underlying SDK client configured for
  Responses API.
- `_pool` (Any | None): lazy PostgreSQL connection pool from infopypg.

Public methods and flow
-----------------------
1. `create_request(**data)` → xAIRequest
   Converts a generic dictionary into a fully validated xAIRequest object.

2. `generate(request, stream=False)` → dict | AsyncIterator
   - Persists request (if `save_mode == "postgres"`).
   - Builds and sends messages using the modern Responses API.
   - For non-streaming: calls `_xai_generate`, persists response.
   - For streaming: calls `_generate_stream_and_persist` (accumulates text
     and persists one final row on completion).

3. `create_batch`, `add_to_batch`, `get_batch_status`,
   `retrieve_and_persist_batch_results`
   - Batch lifecycle with automatic per-item request/response persistence
     using `batch_id` and `batch_index` for ordering and identification.

Persistence flow (when `save_mode == "postgres"`)
-------------------------------------------------
- `_persist_request` → inserts into `requests` table (returns `request_id`,
  `tstamp`).
- `_persist_response` → inserts into `responses` table with composite FK to
  the original request and optional `batch_id`.
- `_persist_batch_requests` / `_persist_batch_results` handle bulk cases
  using `batch_id` in JSONB metadata for easy grouping.

Error handling
--------------
Every path logs a structured message via `self.logger` before raising a
`xAI*Error` subclass (or wrapped SDK exception).

Notes
-----
- All database operations use `infopypg` helpers (`execute_query`,
  `ensure_partition_exists`) and respect daily RANGE partitioning.
- Reasoning content is automatically extracted via `xAIResponse.from_dict`
  when `include_reasoning=True`.
- The module is fully type-annotated for Python 3.12 and passes Pyrefly
  static checking.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import urllib.request
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.error import URLError

from infopypg import ResolvedSettingsDict
from logger import Logger
from pydantic import BaseModel
from xai_sdk import AsyncClient
from xai_sdk.chat import file, image, system, user

from ..data_structures.xai_objects import (
    JSON_INSTRUCTION,
    LLMStreamingChunkProtocol,
    xAIInput,
    xAIJSONResponseSpec,
    xAIMessage,
    xAIRequest,
    xAIResponse,
)
from .xai.xai_batches import *
from .xai.xai_errors import *
from .xai.xai_media import save_media_files
from .xai.xai_persistence import *
from .xai.xai_response_struct import *
from .xai.xai_stream import *

REASONING_MODELS: set[str] = {"grok-4.20-reasoning", "grok-4", "grok-beta"}


class xAIClient:
    """Dedicated xAI client with full error and logging integration."""

    # ------------------------------------------------------------------
    # Batch processing (with Pyrefly type-ignore annotations)
    # see xai/xai_batches.py
    # ------------------------------------------------------------------
    create_batch = create_batch
    add_to_batch = add_to_batch
    get_batch_status = get_batch_status
    retrieve_batch_results = retrieve_batch_results
    retrieve_and_persist_batch_results = retrieve_and_persist_batch_results

    # ------------------------------------------------------------------
    # structured response setup
    # see xai/xai_response_struct.py
    # ------------------------------------------------------------------
    create_json_response_spec = create_json_response_spec
    generate_structured = generate_structured
    generate_structured_stream = generate_structured_stream

    # ------------------------------------------------------------------
    # Streaming response
    # see xai/xai_stream.py
    # ------------------------------------------------------------------
    _generate_stream_and_persist = generate_stream_and_persist
    _xai_stream = xai_stream

    # ------------------------------------------------------------------
    # Postgres persistence (infopypg only)
    # see xai/xai_persistence.py
    # ------------------------------------------------------------------
    _get_pool = get_pool
    _get_or_create_provider_id = get_or_create_provider_id
    _persist_request = persist_request
    _persist_response = persist_response
    _persist_batch_results = persist_batch_results
    _persist_batch_requests = persist_batch_requests

    # ------------------------------------------------------------------
    # Media processing
    # see xai/xai_media.py
    # (This must be present as it is used by the persist_response
    # import.)
    # ------------------------------------------------------------------
    _save_media_files = save_media_files

    # initialise the object
    def __init__(
        self,
        logger: Logger,
        api_key: str | None = None,
        pg_resolved_settings: ResolvedSettingsDict | None = None,
        settings: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        media_root: str = "media",
    ) -> None:
        """Initialise the xAIClient instance.

        Parameters
        ----------
        logger : Logger
            Structured logger (mandatory for all operations).
        api_key : str | None
            xAI API key. Falls back to XAI_API_KEY environment variable if
            None.
        pg_resolved_settings : ResolvedSettingsDict | None
            Resolved PostgreSQL settings for infopypg persistence.
        settings : dict[str, Any] | None
            Additional client configuration dictionary.
        conversation_id : str | None
            Conversation identifier used for prompt caching via x-grok-conv-id
            metadata.
        media_root : str
            Root directory for downloaded media files (default "media").

        Raises
        ------
        xAIClientError
            If the API key is missing or initialisation fails.

        Notes
        -----
        The underlying AsyncClient is created with optional metadata for
        prompt-caching server affinity. The PostgreSQL pool is initialised
        lazily on first use.
        """
        try:
            self.api_key = api_key or os.getenv("XAI_API_KEY")
            if not self.api_key:
                logger.error(
                    "Missing XAI API key",
                    extra={"obj": {"conversation_id": conversation_id}},
                )
                raise xAIClientError("api_key is required for xAIClient")
            self.logger = logger
            self.settings = settings or {}
            self.conversation_id = conversation_id
            metadata = (
                (("x-grok-conv-id", self.conversation_id),)
                if self.conversation_id
                else None
            )
            self._client = AsyncClient(api_key=self.api_key, metadata=metadata)
            self._pg_resolved_settings = pg_resolved_settings
            self._pool = None
            self.media_root: Path = Path(media_root).resolve()
            self.logger.info(
                "xAIClient initialised successfully",
                extra={
                    "obj": {
                        "conversation_id": conversation_id,
                        "media_root": str(self.media_root),
                    }
                },
            )

        except Exception as exc:                                                          # Defensive top-level catch
            if logger:
                logger.error(
                    "xAIClient initialisation failed",
                    extra={"obj": {"error": str(exc)}},
                )
            raise xAIClientError("Failed to initialise xAIClient") from exc

    def create_request(self, **data: Any) -> xAIRequest:
        """Convert generic dictionary into a validated xAIRequest object.

        Parameters
        ----------
        **data : Any
            Arbitrary keyword arguments that will be passed to
            xAIRequest.from_dict.

        Returns
        -------
        xAIRequest
            Fully validated request object.

        Raises
        ------
        xAIClientError
            If the supplied data is invalid (wrapped ValueError from the data
            layer).
        """
        try:
            return xAIRequest.from_dict(data)
        except ValueError as exc:
            self.logger.error(
                "Invalid request data provided to create_request",
                extra={"obj": {"data_keys": list(data.keys()), "error": str(exc)}},
            )
            raise xAIClientError("Invalid request data") from exc

    async def _append_messages_to_chat(self, chat: Any, request: xAIRequest) -> None:
        """Append all messages from a xAIRequest to an SDK chat object.

        Handles system, user, assistant, and developer roles plus multimodal
        content. Extracted to eliminate duplication between generate() and
        batch paths.

        Parameters
        ----------
        chat : Any
            xAI SDK chat instance.
        request : xAIRequest
            Source of messages.
        """
        for msg_dict in request.to_sdk_messages():
            role = msg_dict["role"]
            content = msg_dict["content"]
            if role == "system":
                chat.append(system(content))
            elif role == "user":
                if isinstance(content, str):
                    chat.append(user(content))
                elif isinstance(content, list):
                    parts = []
                    for part in content:
                        if part.get("type") == "input_text":
                            parts.append(part["text"])
                        elif part.get("type") == "input_image":
                            parts.append(image(part["image_url"]))
                        elif part.get("type") == "input_file":
                            parts.append(file(part.get("file_url")))                      # placeholder
                        else:
                            self.logger.error(
                                "Unsupported multimodal content type",
                                extra={"obj": {"part": part}},
                            )
                            raise xAIClientMultimodalError(
                                "Unsupported content type in multimodal message"
                            ) from None
                    chat.append(user(*parts))
            else:                                                                         # assistant or developer
                chat.append(user(content))                                                # SDK treats as continuation

    async def generate(
        self, request: xAIRequest, stream: bool = False, **kwargs: Any
    ) -> dict[str, Any] | AsyncIterator[LLMStreamingChunkProtocol]:
        """Main generation entry point with full parameter forwarding.

        Persists request/response when requested, builds chat with all
        xAIRequest fields (including tools), and returns native result.
        """
        if request.include_reasoning and request.model not in REASONING_MODELS:
            self.logger.error(
                "Unsupported thinking mode requested",
                extra={"obj": {"model": request.model, "include_reasoning": True}},
            )
            raise UnsupportedThinkingModeError(
                f"Reasoning mode not supported for model {request.model}"
            ) from None

        self.logger.info(
            "xAI generation started",
            extra={
                "obj": {
                    "model": request.model,
                    "stream": stream,
                    "cache_key": request.prompt_cache_key,
                }
            },
        )

        persist_info: dict | None = None
        if request.save_mode == "postgres":
            request_id, request_tstamp = await self._persist_request(request)
            persist_info = {"request_id": request_id, "request_tstamp": request_tstamp}

        try:
            chat_kwargs = request.to_chat_create_kwargs()
            chat = self._client.chat.create(**chat_kwargs)
            await self._append_messages_to_chat(chat, request)

            if stream:
                result = (
                    self._generate_stream_and_persist(chat, request, persist_info)
                    if persist_info
                    else self._xai_stream(chat, request)
                )
            else:
                result = await self._xai_generate(chat, request)
                if persist_info and isinstance(result, dict):
                    await self._persist_response(
                        persist_info["request_id"],
                        persist_info["request_tstamp"],
                        result,
                        request,
                    )

            self.logger.info(
                "xAI generation completed",
                extra={"obj": {"save_mode": request.save_mode}},
            )
            return result

        except Exception as exc:                                                          # Catch SDK / network / any unexpected error
            if isinstance(exc, (ConnectionError, TimeoutError)):
                self.logger.error(
                    "xAI API connection failure", extra={"obj": {"error": str(exc)}}
                )
                raise xAIAPIConnectionError("Failed to connect to xAI API") from exc
            if (
                "unauthenticated" in str(exc).lower()
                or "invalid api key" in str(exc).lower()
            ):
                self.logger.error(
                    "xAI API authentication failure",
                    extra={"obj": {"error": str(exc)}},
                )
                raise xAIAPIAuthenticationError(
                    "Invalid or missing XAI_API_KEY"
                ) from exc
            if "rate limit" in str(exc).lower():
                self.logger.error(
                    "xAI API rate limit exceeded", extra={"obj": {"error": str(exc)}}
                )
                raise xAIAPIRateLimitError("Rate limit exceeded") from exc

            self.logger.error(
                "Unexpected error during xAI generation",
                extra={"obj": {"error": str(exc), "model": request.model}},
            )
            raise wrap_xai_api_error(exc, "xAI API call failed") from exc

    async def _xai_generate(self, chat: Any, request: xAIRequest) -> dict[str, Any]:
        """Non-streaming helper (wrapped for logging)."""
        try:
            sdk_response = await chat.sample()
            return {
                "output": getattr(sdk_response, "content", str(sdk_response)),
                "model": request.model,
                "finish_reason": getattr(sdk_response, "finish_reason", None),
                "raw": getattr(
                    sdk_response,
                    "proto",
                    sdk_response.__dict__ if hasattr(sdk_response, "__dict__") else {},
                ),
            }
        except Exception as exc:
            self.logger.error(
                "Non-streaming generation failed", extra={"obj": {"error": str(exc)}}
            )
            raise wrap_xai_api_error(exc, "Non-streaming generation failed") from exc

    def _build_endpoint(self, request: xAIRequest) -> dict[str, Any]:
        """Consistent endpoint metadata – now using the modern Responses API."""
        return {
            "provider": "xai",
            "model": request.model,
            "host": "api.x.ai",
            "endpoint_path": "/v1/responses",                                             # Updated for Responses API
            "prompt_cache_key": request.prompt_cache_key,
        }
