import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from aioresponses import aioresponses

from ai_api import create


@pytest.fixture(scope="session")
def event_loop_policy():
    """Required for pytest-asyncio on Python 3.12+"""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def mock_aiohttp():
    """Mock all aiohttp calls cleanly"""
    with aioresponses() as m:
        yield m


@pytest.fixture
async def grok_client(mock_aiohttp):
    """Fully mocked Grok client (no real network, no real DB)"""
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
                        save_mode="none",                                                 # fastest for unit tests
                        concurrency=5,
                        max_retries=2,
                    )
                    return client
