"""Shared low-level utilities for all xAI chat modalities."""

import json
from typing import Any, AsyncIterator

import httpx

from ...data_structures.xai_objects import xAIRequest
from ..xai_client import BaseXAIClient                                                    # for type reference only
from .errors_xai import wrap_xai_api_error, xAIAPIError, xAIClientError

__all__: list[str] = [
    "_build_endpoint",
    "_generate_non_streaming",
    "_generate_streaming",
]


def _build_endpoint(self, request: xAIRequest) -> dict[str, Any]:
    """Consistent endpoint metadata – now using the modern Responses API."""
    return {
        "provider": "xai",
        "model": request.model,
        "host": "api.x.ai",
        "endpoint_path": "/v1/responses",                                                 # Updated for Responses API
        "prompt_cache_key": request.prompt_cache_key,
    }


async def _generate_non_streaming(
    client: BaseXAIClient,
    json_data: dict[str, Any],
    endpoint: str = "/v1/responses",
) -> dict[str, Any]:
    """Non-streaming (turn, batch, multimodal) HTTP POST to the xAI Responses API."""
    url = f"{client.base_url}{endpoint}"
    headers = {
        "Authorization": f"Bearer {client.api_key}",
        "Content-Type": "application/json",
    }

    client.logger.debug(
        "xAI non-streaming request initiated",
        extra={"obj": {"endpoint": endpoint, "model": json_data.get("model")}},
    )

    try:
        response = await client._http_client.post(
            url,
            json=json_data,
            headers=headers,
            timeout=client.timeout,
        )
        response.raise_for_status()

        client.logger.info(
            "xAI non-streaming request succeeded",
            extra={"obj": {"endpoint": endpoint, "status_code": response.status_code}},
        )

        return response.json()

    except httpx.HTTPStatusError as exc:
        wrapped = wrap_xai_api_error(
            exc, f"xAI API returned HTTP {exc.response.status_code}"
        )
        client.logger.error(
            "xAI API error (non-streaming)",
            extra={
                "obj": {
                    "status_code": exc.response.status_code,
                    "url": str(exc.request.url),
                }
            },
        )
        raise wrapped from exc

    except httpx.RequestError as exc:
        wrapped = wrap_xai_api_error(exc, f"HTTP request to xAI failed: {exc}")
        client.logger.error(
            "xAI HTTP transport error (non-streaming)",
            extra={"obj": {"endpoint": endpoint}},
        )
        raise wrapped from exc

    except Exception as exc:
        client.logger.error(
            "Unexpected error in non-streaming request",
            extra={"obj": {"endpoint": endpoint}},
        )
        raise xAIClientError(
            f"Non-streaming request to {endpoint} failed: {exc}"
        ) from exc


async def _generate_streaming(
    client: BaseXAIClient,
    json_data: dict[str, Any],
    endpoint: str = "/v1/responses",
) -> AsyncIterator[dict[str, Any]]:
    """Streaming SSE generator for the xAI Responses API."""
    url = f"{client.base_url}{endpoint}"
    headers = {
        "Authorization": f"Bearer {client.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    client.logger.debug(
        "xAI streaming request initiated",
        extra={"obj": {"endpoint": endpoint, "model": json_data.get("model")}},
    )

    try:
        async with client._http_client.stream(
            "POST",
            url,
            json=json_data,
            headers=headers,
            timeout=client.timeout,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break

                try:
                    data = json.loads(line[6:])                                           # strip "data: "
                    yield data
                except json.JSONDecodeError:
                    client.logger.warning(
                        "Malformed SSE chunk received",
                        extra={"obj": {"line": line}},
                    )
                    continue

        client.logger.info(
            "xAI streaming request completed", extra={"obj": {"endpoint": endpoint}}
        )

    except httpx.HTTPStatusError as exc:
        wrapped = wrap_xai_api_error(
            exc, f"xAI API returned HTTP {exc.response.status_code}"
        )
        client.logger.error(
            "xAI API error (streaming)",
            extra={
                "obj": {
                    "status_code": exc.response.status_code,
                    "url": str(exc.request.url),
                }
            },
        )
        raise wrapped from exc

    except httpx.RequestError as exc:
        wrapped = wrap_xai_api_error(exc, f"HTTP request to xAI failed: {exc}")
        client.logger.error(
            "xAI HTTP transport error (streaming)",
            extra={"obj": {"endpoint": endpoint}},
        )
        raise wrapped from exc

    except Exception as exc:
        client.logger.error(
            "Unexpected error in streaming request",
            extra={"obj": {"endpoint": endpoint}},
        )
        raise xAIClientError(f"Streaming request to {endpoint} failed: {exc}") from exc
