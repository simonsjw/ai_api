import pytest

from ai_api import create


@pytest.mark.asyncio
async def test_create_grok():
    client = await create(provider="grok", model="grok-2", api_key="dummy")
    assert client.provider_name == "grok"
    assert client.model == "grok-2"
    assert client.can_stream() is True


@pytest.mark.asyncio
async def test_create_ollama():
    client = await create(
        provider="ollama", model="qwen3-coder-next:latest", org="localhost"
    )
    assert client.provider_name == "ollama"
