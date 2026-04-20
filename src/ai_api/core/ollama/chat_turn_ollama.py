"""Turn-based (non-streaming) chat functionality for the Ollama native API.

Creates a single non-streaming request to /api/chat.
Fully integrated with:
- ollama_objects.py (OllamaRequest / OllamaResponse / OllamaJSONResponseSpec)
- xAIPersistenceManager (reused unchanged)
- structured JSON output support
- consistent logging, persistence, and error style from the xAI side

Key transparent differences from chat_turn_xai.py (all documented inline):
- No xAI SDK → plain httpx.AsyncClient calling /api/chat
- Uses "messages" (Ollama) instead of xAI's "input" / SDK chat object
- Ollama response shape includes rich local telemetry (durations, token counts)
- Structured output uses Ollama's "format" key (handled automatically in OllamaRequest)
"""

import uuid
from datetime import datetime
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from ...data_structures.ollama_objects import (
    OllamaInput,
    OllamaJSONResponseSpec,
    OllamaRequest,
    OllamaResponse,
    SaveMode,
)
from ..ollama_client import BaseOllamaClient
from .errors_ollama import OllamaClientError, wrap_ollama_error

__all__: list[str] = ["create_turn_chat_session"]


async def create_turn_chat_session(
    client: "BaseOllamaClient",                                                           # forward reference – defined in ollama_client.py
    messages: list[dict[str, Any]],
    model: str = "llama3.2",
    *,
    response_model: type[BaseModel] | None = None,                                        # structured output
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    save_mode: SaveMode = "none",
    **kwargs: Any,
) -> OllamaResponse:
    """Create a stateful turn-based chat using Ollama's native /api/chat endpoint.

    Mirrors create_turn_chat_session from the xAI side as closely as possible.
    Fully supports multimodal (base64 images) via OllamaMessage.
    Media files are automatically saved when persistence_manager.media_root is set.
    """
    client.logger.info(
        "Creating turn-based Ollama chat (native API)",
        extra={
            "model": model,
            "message_count": len(messages),
            "has_media": any(
                msg.get("images")
                or (
                    isinstance(msg.get("content"), list)
                    and any(
                        isinstance(part, dict) and part.get("images")
                        for part in msg.get("content", [])
                    )
                )
                for msg in messages
            ),
        },
    )

    # 1. Build canonical OllamaRequest (reuses all validation + helpers from ollama_objects.py)
    ollama_input = OllamaInput.from_list(messages)
    request = OllamaRequest(
        model=model,
        input=ollama_input,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        response_format=OllamaJSONResponseSpec(model=response_model)
        if response_model
        else None,
        save_mode=save_mode,
        **{k: v for k, v in kwargs.items() if k not in {"response_model", "save_mode"}},
    )

    # 2. Persist request BEFORE network call (exact same contract as xAI)
    request_id: uuid.UUID | None = None
    request_tstamp: datetime | None = None
    if save_mode == "postgres" and client.persistence_manager is not None:
        try:
            (
                request_id,
                request_tstamp,
            ) = await client.persistence_manager.persist_request(request)
        except Exception as exc:                                                          # noqa: BLE001
            client.logger.warning(
                "Request persistence failed (continuing)",
                extra={"error": str(exc)},
            )

            # 3. HTTP call to Ollama /api/chat (non-streaming)
    http_client: httpx.AsyncClient = await client._get_http_client()                      # type: ignore[attr-defined]
    payload = request.to_ollama_dict()
    payload["stream"] = False

    try:
        resp = await http_client.post("/api/chat", json=payload)
        resp.raise_for_status()
        raw_data = resp.json()
    except Exception as exc:
        client.logger.error(
            "Ollama turn-based request failed", extra={"error": str(exc)}
        )
        raise wrap_ollama_error(
            exc, "Failed to generate turn-based response via Ollama"
        ) from exc

    # 4. Convert raw JSON → canonical OllamaResponse
    response: OllamaResponse = OllamaResponse.from_dict(raw_data)

    # 5. Structured output handling (exactly like response_struct_xai.py)
    if response_model is not None:
        try:
            parsed = response_model.model_validate_json(response.text)
            response.parsed = parsed
        except Exception as parse_exc:                                                    # noqa: BLE001
            client.logger.warning(
                "Structured output parsing failed",
                extra={"error": str(parse_exc)},
            )

            # 6. Persist response (exact same pattern as xAI)
    if (
        save_mode == "postgres"
        and client.persistence_manager is not None
        and request_id is not None
        and request_tstamp is not None
    ):
        try:
            await client.persistence_manager.persist_response(
                request_id=request_id,
                request_tstamp=request_tstamp,
                api_result={"raw": raw_data, "model": model},
                request=request,
            )
        except Exception as exc:                                                          # noqa: BLE001
            client.logger.warning(
                "Response persistence failed (continuing)",
                extra={"error": str(exc)},
            )

    client.logger.info(
        "Turn-based Ollama chat completed successfully",
        extra={"model": model},
    )
    return response
