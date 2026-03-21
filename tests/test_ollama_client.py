"""
Tests for the Ollama concrete client.
Fully mocked — no real network or Ollama server required.
"""

import json

import pytest

from ai_api.core.request import LLMRequest
from ai_api.data_structures.LLM_types_ollama import (
    OllamaInput,                                                                          # adjust import if your type is elsewhere
)


@pytest.mark.asyncio
async def test_create_ollama(ollama_client, mock_aiohttp):
    """Basic factory + provider verification (uses the fixture from conftest)."""
    assert ollama_client.provider_name == "ollama"
    assert ollama_client.model == "llama3.2"
    assert ollama_client.can_stream() is True


@pytest.mark.asyncio
async def test_submit_batch_success_ollama(ollama_client, mock_aiohttp):
    """Mock a successful Ollama chat response."""
    mock_aiohttp.post(
        "http://localhost:11434/api/chat",                                                # default Ollama endpoint in the client
        status=200,
        body=json.dumps(
            {
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "Hello from Ollama!"},
                "done": True,
            }
        ),
        headers={"content-type": "application/json"},
    )

    messages = [{"role": "user", "content": "Hi!"}]
    req = LLMRequest(input=OllamaInput.from_list(messages), model="llama3.2")

    responses = await ollama_client.submit_batch([req])
    assert len(responses) == 1
    assert responses[0].content == "Hello from Ollama!"
    assert responses[0].model == "llama3.2"


@pytest.mark.asyncio
async def test_streaming_ollama(ollama_client, mock_aiohttp):
    """Test token-by-token streaming (Ollama SSE format)."""
    mock_aiohttp.post(
        "http://localhost:11434/api/chat",
        status=200,
        body=b'data: {"message":{"content":"Hello"}}\n\n'
        b'data: {"message":{"content":" from Ollama!"}}\n\n'
        b'data: {"done":true}\n',
        headers={"content-type": "application/x-ndjson"},
    )

    req = LLMRequest(...)                                                                 # same as above
    chunks = []
    async for chunk in ollama_client.stream(req):
        chunks.append(chunk.delta_text)

    assert "".join(chunks) == "Hello from Ollama!"


@pytest.mark.asyncio
async def test_ollama_custom_host(ollama_client_factory):
    """Test custom org/host works (factory helper if you added one to conftest)."""
    client = await ollama_client_factory(org="192.168.1.100:11434")
    assert "192.168.1.100:11434" in str(client.base_url)
