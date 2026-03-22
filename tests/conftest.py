import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from aioresponses import aioresponses

from ai_api import create


@pytest_asyncio.fixture
async def mock_aiohttp():
    """Mock all aiohttp calls."""
    with aioresponses() as m:
        yield m


@pytest_asyncio.fixture
async def grok_client(mock_aiohttp):
    """Fully mocked Grok client (no real network or DB)."""
    with patch(
        "ai_api.clients.grok.client.GrokConcreteClient._get_provider_id",
        new=AsyncMock(return_value=1),
    ):
        with patch(
            "ai_api.clients.grok.client.GrokConcreteClient._get_pool", new=AsyncMock()
        ):
            with patch(
                "ai_api.clients.grok.client.GrokConcreteClient._save_request_to_postgres",
                new=AsyncMock(),
            ):
                with patch(
                    "ai_api.clients.grok.client.GrokConcreteClient._save_response_to_postgres",
                    new=AsyncMock(),
                ):
                    client = await create(
                        provider="grok",
                        model="grok-2",
                        api_key="xai-test-key",
                        save_mode="none",
                        concurrency=5,
                        max_retries=2,
                    )
                    return client


@pytest_asyncio.fixture
async def ollama_client(mock_aiohttp):
    """Mocked Ollama client — model existence check disabled for tests."""
    with patch(
        "ai_api.clients.ollama.client.OllamaConcreteClient._ensure_model_exists",
        new=AsyncMock(),
    ):
        client = await create(
            provider="ollama",
            model="llama3.2",
            org="localhost",
            save_mode="none",
            concurrency=5,
            max_retries=2,
        )
        return client


@pytest_asyncio.fixture
async def ollama_client_with_reasoning(mock_aiohttp):
    """Ollama client for reasoning tests — model check disabled."""
    with patch(
        "ai_api.clients.ollama.client.OllamaConcreteClient._ensure_model_exists",
        new=AsyncMock(),
    ):
        client = await create(
            provider="ollama",
            model="deepseek-r1",
            org="localhost",
            save_mode="none",
        )
        return client
