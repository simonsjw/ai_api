"""Streaming chat functionality for the xAI API using the modern Responses API."""

import uuid
from datetime import datetime
from typing import Any, AsyncIterator, Optional

from ...data_structures.xai_objects import (
    LLMStreamingChunkProtocol,
    SaveMode,
    xAIInput,
    xAIRequest,
    xAIResponse,
    xAIStreamingChunk,
)
from ..xai_client import BaseXAIClient
from .common_xai import _generate_streaming
from .errors_xai import wrap_infopypg_error, xAIClientError
from .persistence_xai import xAIPersistenceManager


async def create_stream_chat(
    client: BaseXAIClient,
    messages: list[dict[str, Any]],
    model: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    save_mode: SaveMode = "none",
    **kwargs: Any,
) -> AsyncIterator[tuple[xAIResponse, LLMStreamingChunkProtocol]]:
    """Create a streaming chat completion using the Responses API.

    Yields (accumulating_response, chunk) pairs for real-time output.
    The accumulating xAIResponse is updated on every chunk.
    Persistence (if save_mode="postgres") captures the request before streaming
    and the final complete response once the stream ends.
    """
    # 1. Build the canonical request object
    xai_input = xAIInput.from_list(messages)
    request = xAIRequest(
        input=xai_input,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        save_mode=save_mode,
        **kwargs,
    )

    # 2. Prepare the exact kwargs expected by the xAI Responses API
    request_kwargs = request.to_api_kwargs()
    # Ensure streaming is enabled (Responses API requires explicit "stream": true)
    request_kwargs["stream"] = True

    request_id: uuid.UUID | None = None
    request_tstamp: datetime | None = None
    final_raw_response: dict[str, Any] | None = None

    client.logger.info(
        "Starting xAI streaming chat",
        extra={"obj": {"model": model, "save_mode": save_mode}},
    )

    # === PERSISTENCE: REQUEST (before streaming starts) ===
    if save_mode == "postgres" and client.persistence_manager is not None:
        try:
            (
                request_id,
                request_tstamp,
            ) = await client.persistence_manager.persist_request(request)
        except Exception as exc:
            client.logger.warning(
                "Request persistence failed for streaming chat (continuing with API call)",
                extra={"obj": {"model": model, "error": str(exc)}},
            )
            raise wrap_infopypg_error(
                exc, "Failed to persist streaming request"
            ) from exc

    try:
        accumulated: xAIResponse | None = None

        # Low-level streaming generator (delegated to common_xai._generate)
        async for raw_chunk in _generate_streaming(
            client=client,
            json_data=request_kwargs,
            endpoint="/v1/responses",
        ):
            # Convert raw SSE chunk into domain objects
            chunk = xAIStreamingChunk(
                text=raw_chunk.get("delta", {}).get("content", "") or "",
                finish_reason=raw_chunk.get("finish_reason"),
                tool_calls_delta=raw_chunk.get("tool_calls"),
                is_final=raw_chunk.get("finish_reason") is not None,
                raw=raw_chunk,
            )

            # Build / update the accumulating response object
            if accumulated is None:
                accumulated = xAIResponse.from_dict(raw_chunk)
            else:
                # Merge the latest chunk into the accumulating response
                accumulated = xAIResponse.from_dict(raw_chunk)

            yield accumulated, chunk

            # Detect end of stream
            if chunk.finish_reason is not None:
                final_raw_response = raw_chunk
                break

            # === PERSISTENCE: FINAL RESPONSE (once stream completes) ===
        if (
            save_mode == "postgres"
            and client.persistence_manager is not None
            and request_id is not None
            and request_tstamp is not None
            and final_raw_response is not None
        ):
            await client.persistence_manager.persist_response(
                request_id=request_id,
                request_tstamp=request_tstamp,
                api_result=final_raw_response,
                request=request,
            )

        client.logger.info(
            "Streaming xAI chat completed successfully",
            extra={"obj": {"model": model}},
        )

    except Exception as exc:
        client.logger.error(
            "Streaming chat creation failed",
            extra={"obj": {"model": model}},
        )
        raise xAIClientError(f"Streaming chat creation failed: {exc}") from exc
