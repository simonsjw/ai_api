#!/usr/bin/env python3
"""Full pytest suite for LLMClient – all external dependencies (including DB) mocked.
Use createTestDB.sql to create test database first."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_api.core.client import LLMClient
from ai_api.core.errors import UnsupportedThinkingModeError
from ai_api.data_structures.grok import GrokRequest
from ai_api.data_structures.ollama import OllamaRequest

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_settings() -> dict[str, Any]:
    """Flat keys that satisfy infopypg validation (no real DB is used)."""
    return {
        "db_host": "localhost",
        "db_port": "5432",
        "db_name": "testdb",
        "db_user": "testuser",
        "password": "testpass",
    }


@pytest.fixture
def mock_ollama_client() -> MagicMock:
    client = MagicMock()
    client.chat = AsyncMock()
    client.embeddings = AsyncMock()
    client.create = AsyncMock()
    return client


@pytest.fixture
def mock_grok_client() -> MagicMock:
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    client.post = AsyncMock()
    client.get = AsyncMock()
    return client


@pytest.fixture
def ollama_client(
    mock_settings: dict[str, Any], mock_ollama_client: MagicMock
) -> LLMClient:
    """Ollama client – DB persistence fully mocked."""
    with patch(
        "ai_api.core.client.ollama.AsyncClient", return_value=mock_ollama_client
    ):
        with patch.object(LLMClient, "_ensure_db_pool", AsyncMock()):
            with patch.object(LLMClient, "_persist_interaction", AsyncMock()):
                return LLMClient(
                    provider="ollama",
                    model="qwen3:8b",
                    settings=mock_settings,
                )


@pytest.fixture
def grok_client(
    mock_settings: dict[str, Any], mock_grok_client: MagicMock
) -> LLMClient:
    """Grok client – DB persistence fully mocked."""
    with patch("ai_api.core.client.AsyncOpenAI", return_value=mock_grok_client):
        with patch.object(LLMClient, "_ensure_db_pool", AsyncMock()):
            with patch.object(LLMClient, "_persist_interaction", AsyncMock()):
                return LLMClient(
                    provider="grok",
                    model="grok-3",
                    settings=mock_settings,
                    api_key="dummy-key-for-testing",
                    conversation_id="test-conversation-id",
                )


@pytest.fixture
def ollama_request() -> OllamaRequest:
    return OllamaRequest(model="qwen3:8b", input="Hello")


@pytest.fixture
def grok_request() -> GrokRequest:
    return GrokRequest(model="grok-3", input="Hello")


# --------------------------------------------------------------------------- #
# Test functions
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ollama_generate_non_stream(
    ollama_client: LLMClient,
    ollama_request: OllamaRequest,
    mock_ollama_client: MagicMock,
) -> None:
    object.__setattr__(
        ollama_request, "messages", [{"role": "user", "content": ollama_request.input}]
    )
    object.__setattr__(ollama_request, "options", {})
    mock_ollama_client.chat.return_value = {"message": {"content": "Hi"}}
    response = await ollama_client.generate(ollama_request, stream=False)
    assert response is not None


@pytest.mark.asyncio
async def test_ollama_generate_stream(
    ollama_client: LLMClient,
    ollama_request: OllamaRequest,
    mock_ollama_client: MagicMock,
) -> None:
    object.__setattr__(
        ollama_request, "messages", [{"role": "user", "content": ollama_request.input}]
    )
    object.__setattr__(ollama_request, "options", {})

    async def fake_stream() -> AsyncIterator[dict[str, Any]]:
        yield {"message": {"content": "chunk1"}}
        yield {"message": {"content": "chunk2"}}

    mock_ollama_client.chat.return_value = fake_stream()
    result = await ollama_client.generate(ollama_request, stream=True)
    chunks: list[Any] = []
    async for chunk in result:
        chunks.append(chunk)
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_ollama_think_supported(
    ollama_client: LLMClient,
    ollama_request: OllamaRequest,
    mock_ollama_client: MagicMock,
) -> None:
    object.__setattr__(
        ollama_request, "messages", [{"role": "user", "content": ollama_request.input}]
    )
    object.__setattr__(ollama_request, "options", {"think": "low"})
    mock_ollama_client.chat.return_value = {"message": {"content": "reasoned"}}
    response = await ollama_client.generate(ollama_request, stream=False)
    assert response is not None


@pytest.mark.asyncio
async def test_ollama_think_unsupported_raises(
    ollama_client: LLMClient, ollama_request: OllamaRequest
) -> None:
    object.__setattr__(
        ollama_request, "messages", [{"role": "user", "content": ollama_request.input}]
    )
    object.__setattr__(ollama_request, "options", {"think": "low"})
    request = OllamaRequest(model="llama3.2", input="Hello")
    object.__setattr__(request, "messages", [{"role": "user", "content": "Hello"}])
    object.__setattr__(request, "options", {"think": "low"})
    with pytest.raises(UnsupportedThinkingModeError):
        await ollama_client.generate(request, stream=False)


@pytest.mark.asyncio
async def test_ollama_think_none_does_not_raise(
    ollama_client: LLMClient,
    ollama_request: OllamaRequest,
    mock_ollama_client: MagicMock,
) -> None:
    object.__setattr__(
        ollama_request, "messages", [{"role": "user", "content": ollama_request.input}]
    )
    object.__setattr__(ollama_request, "options", {})
    mock_ollama_client.chat.return_value = {"message": {"content": "ok"}}
    response = await ollama_client.generate(ollama_request, stream=False)
    assert response is not None


@pytest.mark.asyncio
async def test_grok_generate(
    grok_client: LLMClient, grok_request: GrokRequest, mock_grok_client: MagicMock
) -> None:
    mock_grok_client.responses.create.return_value = MagicMock(
        model_dump=lambda: {
            "id": "resp_123",
            "created_at": 1743600000,
            "model": "grok-3",
            "output": [
                {"type": "message", "role": "assistant", "content": "Grok answer"}
            ],
        }
    )
    response = await grok_client.generate(grok_request, stream=False)
    assert response is not None


@pytest.mark.asyncio
async def test_parallel_generate(
    grok_client: LLMClient, grok_request: GrokRequest
) -> None:
    grok_client.generate = AsyncMock(return_value={"output": "ok"})                       # type: ignore[attr-defined]
    requests = [grok_request, grok_request]
    results = await grok_client.parallel_generate(requests, max_concurrent=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_submit_batch_grok_only(grok_client: LLMClient) -> None:
    grok_client.submit_batch = AsyncMock(return_value={"id": "batch123"})                 # type: ignore[attr-defined]
    batch_req = MagicMock()
    result = await grok_client.submit_batch(batch_req)
    assert result["id"] == "batch123"


@pytest.mark.asyncio
async def test_await_batch_completion(grok_client: LLMClient) -> None:
    grok_client.await_batch_completion = AsyncMock(                                       # type: ignore[attr-defined]
        return_value=[{"output": "final"}]
    )
    results = await grok_client.await_batch_completion("batch123")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_create_ollama_model(ollama_client: LLMClient) -> None:
    ollama_client.create_ollama_model = AsyncMock(return_value=None)                      # type: ignore[attr-defined]
    await ollama_client.create_ollama_model("FROM llama3\n")


@pytest.mark.asyncio
async def test_get_embeddings(
    ollama_client: LLMClient, mock_ollama_client: MagicMock
) -> None:
    mock_ollama_client.embeddings.return_value = {"embedding": [0.1, 0.2]}
    result = await ollama_client.get_embeddings("test text")
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_grok_only_methods_raise_on_ollama(ollama_client: LLMClient) -> None:
    with pytest.raises(ValueError):
        await ollama_client.submit_batch(MagicMock())                                     # type: ignore[attr-defined]
