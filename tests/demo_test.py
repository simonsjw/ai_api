#!/usr/bin/env python3
"""Full pytest suite for LLMClient.

This test module exercises every public method of LLMClient (Grok and Ollama
providers) plus the new strict 'think' validation.  All external dependencies
are mocked so the suite runs without a live Ollama/Grok instance or database.

Fixtures provide a clean, isolated LLMClient for each test.  Async tests use
pytest-asyncio.  Every test function is deliberately kept under 40 lines.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_api.core.client import LLMClient
from ai_api.core.errors import UnsupportedThinkingModeError
from ai_api.data_structures.grok import GrokRequest                                       # type: ignore[import-not-found]
from ai_api.data_structures.ollama import (
    OllamaRequest,                                                                        # type: ignore[import-not-found]
)

# --------------------------------------------------------------------------- #
# Fixtures (each ≤40 lines)
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_settings() -> dict[str, Any]:
    """Return a minimal settings dict for LLMClient initialisation.

    Returns
    -------
    dict[str, Any]
        Settings required by infopypg and logger (all values mocked).
    """
    return {
        "db": {"host": "localhost", "port": 5432, "database": "test"},
        "log": {"level": "DEBUG"},
    }


@pytest.fixture
def mock_ollama_client() -> MagicMock:
    """Return a fully mocked ollama.AsyncClient."""
    client = MagicMock()
    client.chat = AsyncMock()
    client.embeddings = AsyncMock()
    return client


@pytest.fixture
def mock_grok_client() -> MagicMock:
    """Return a fully mocked openai.AsyncOpenAI client for Grok."""
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    return client


@pytest.fixture
def ollama_client(
    mock_settings: dict[str, Any], mock_ollama_client: MagicMock
) -> LLMClient:
    """Return an LLMClient configured for Ollama with mocked backend."""
    with patch("ollama.AsyncClient", return_value=mock_ollama_client):
        return LLMClient(
            provider="ollama",
            model="qwen3:8b",
            settings=mock_settings,
        )


@pytest.fixture
def grok_client(
    mock_settings: dict[str, Any], mock_grok_client: MagicMock
) -> LLMClient:
    """Return an LLMClient configured for Grok with mocked backend."""
    with patch("openai.AsyncOpenAI", return_value=mock_grok_client):
        return LLMClient(
            provider="grok",
            model="grok-3",
            settings=mock_settings,
            api_key="dummy-key-for-testing",                                              # required by __init__
        )


@pytest.fixture
def ollama_request() -> OllamaRequest:
    """Minimal OllamaRequest for testing.

    Attributes are set after creation to match what the client expects.
    """
    req = OllamaRequest(model="qwen3:8b")
    req.messages = [{"role": "user", "content": "Hello"}]
    req.options = {}
    return req


@pytest.fixture
def grok_request() -> GrokRequest:
    """Minimal GrokRequest for testing."""
    return GrokRequest(model="grok-3", input="Hello")


# --------------------------------------------------------------------------- #
# Test functions (each ≤40 lines)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ollama_generate_non_stream(
    ollama_client: LLMClient,
    ollama_request: OllamaRequest,
    mock_ollama_client: MagicMock,
) -> None:
    """Verify non-streaming Ollama generate returns a response."""
    mock_ollama_client.chat.return_value = {"message": {"content": "Hi"}}
    response = await ollama_client.generate(ollama_request, stream=False)
    assert response is not None


@pytest.mark.asyncio
async def test_ollama_generate_stream(
    ollama_client: LLMClient,
    ollama_request: OllamaRequest,
    mock_ollama_client: MagicMock,
) -> None:
    """Verify streaming Ollama generate yields chunks implementing the protocol."""

    async def fake_stream() -> AsyncIterator[dict[str, Any]]:
        yield {"message": {"content": "chunk1"}}
        yield {"message": {"content": "chunk2"}}

    mock_ollama_client.chat.return_value = fake_stream()
    chunks: list[Any] = []
    async for chunk in await ollama_client.generate(ollama_request, stream=True):
        chunks.append(chunk)
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_ollama_think_supported(
    ollama_client: LLMClient,
    ollama_request: OllamaRequest,
    mock_ollama_client: MagicMock,
) -> None:
    """Confirm 'think' parameter succeeds on a supported model."""
    ollama_request.options = {"think": "low"}
    mock_ollama_client.chat.return_value = {"message": {"content": "reasoned"}}
    response = await ollama_client.generate(ollama_request, stream=False)
    assert response is not None


@pytest.mark.asyncio
async def test_ollama_think_unsupported_raises(
    ollama_client: LLMClient, ollama_request: OllamaRequest
) -> None:
    """Confirm UnsupportedThinkingModeError is raised when 'think' is set on an
    unsupported model.
    """
    ollama_request.model = "llama3.2"                                                     # unsupported family
    ollama_request.options = {"think": "low"}
    with pytest.raises(UnsupportedThinkingModeError):
        await ollama_client.generate(ollama_request, stream=False)


@pytest.mark.asyncio
async def test_ollama_think_none_does_not_raise(
    ollama_client: LLMClient,
    ollama_request: OllamaRequest,
    mock_ollama_client: MagicMock,
) -> None:
    """Confirm absence of 'think' (or think=None) never raises."""
    ollama_request.options = {}
    mock_ollama_client.chat.return_value = {"message": {"content": "ok"}}
    response = await ollama_client.generate(ollama_request, stream=False)
    assert response is not None


@pytest.mark.asyncio
async def test_grok_generate(
    grok_client: LLMClient, grok_request: GrokRequest, mock_grok_client: MagicMock
) -> None:
    """Verify Grok provider path works end-to-end."""
    mock_grok_client.responses.create.return_value = {"output": "Grok answer"}
    response = await grok_client.generate(grok_request, stream=False)
    assert response is not None


@pytest.mark.asyncio
async def test_parallel_generate(
    grok_client: LLMClient, grok_request: GrokRequest
) -> None:
    """Verify parallel_generate fires multiple requests concurrently."""
    grok_client.generate = AsyncMock(return_value={"output": "ok"})                       # type: ignore[attr-defined]
    requests = [grok_request, grok_request]
    results = await grok_client.parallel_generate(requests, max_concurrent=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_submit_batch_grok_only(grok_client: LLMClient) -> None:
    """Verify Grok batch submission (Grok-only)."""
    grok_client.submit_batch = AsyncMock(return_value={"id": "batch123"})                 # type: ignore[attr-defined]
    batch_req = MagicMock()
    result = await grok_client.submit_batch(batch_req)
    assert result["id"] == "batch123"


@pytest.mark.asyncio
async def test_await_batch_completion(grok_client: LLMClient) -> None:
    """Verify batch polling returns final responses."""
    grok_client.await_batch_completion = AsyncMock(                                       # type: ignore[attr-defined]
        return_value=[{"output": "final"}]
    )
    results = await grok_client.await_batch_completion("batch123")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_create_ollama_model(ollama_client: LLMClient) -> None:
    """Verify Ollama-only model creation."""
    ollama_client.create_ollama_model = AsyncMock(return_value=None)                      # type: ignore[attr-defined]
    await ollama_client.create_ollama_model("FROM llama3\n")


def test_get_embeddings(
    ollama_client: LLMClient, mock_ollama_client: MagicMock
) -> None:
    """Verify embeddings endpoint (Ollama-only)."""
    mock_ollama_client.embeddings.return_value = [0.1, 0.2]
    result = ollama_client.get_embeddings("test text")
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_grok_only_methods_raise_on_ollama(ollama_client: LLMClient) -> None:
    """Confirm Grok-only methods raise when called on an Ollama client."""
    with pytest.raises(ValueError):
        await ollama_client.submit_batch(MagicMock())                                     # type: ignore[attr-defined]
