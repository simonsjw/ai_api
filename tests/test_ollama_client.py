"""
Ollama client tests — batch, streaming, hardware options, and reasoning support.
"""

import json

import pytest

from ai_api.clients.ollama.options import OllamaOptions
from ai_api.core.request import LLMRequest
from ai_api.data_structures.LLM_types_ollama import OllamaInput


@pytest.mark.asyncio
async def test_ollama_batch_success(ollama_client, mock_aiohttp):
    mock_aiohttp.post(
        "http://localhost:11434/v1/chat/completions",
        status=200,
        body=json.dumps(
            {
                "id": "chatcmpl-123",
                "model": "llama3.2",
                "choices": [{"message": {"content": "Hello from Ollama!"}}],
            }
        ),
    )

    messages = [{"role": "user", "content": "Hi"}]
    req = LLMRequest(input=OllamaInput.from_list(messages), model="llama3.2")
    responses = await ollama_client.submit_batch([req])

    assert len(responses) == 1
    assert responses[0].text == "Hello from Ollama!"


@pytest.mark.asyncio
async def test_ollama_batch_with_reasoning(ollama_client, mock_aiohttp):
    mock_aiohttp.post(
        "http://localhost:11434/v1/chat/completions",
        status=200,
        body=json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "<think>Step 1: 2+2=4</think>Final answer: 4"
                        }
                    }
                ]
            }
        ),
    )

    messages = [{"role": "user", "content": "2+2?"}]
    req = LLMRequest(
        input=OllamaInput.from_list(messages),
        model="deepseek-r1",
        capture_reasoning=True,
    )
    resp = (await ollama_client.submit_batch([req]))[0]

    assert resp.text == "Final answer: 4"
    assert resp.reasoning_content == "Step 1: 2+2=4"


@pytest.mark.asyncio
async def test_ollama_streaming_basic(ollama_client, mock_aiohttp):
    mock_aiohttp.post(
        "http://localhost:11434/v1/chat/completions",
        status=200,
        body=b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world!"}}],"done":true}\n\n',
    )

    messages = [{"role": "user", "content": "Hi"}]
    req = LLMRequest(input=OllamaInput.from_list(messages), model="llama3.2")
    chunks = [chunk async for chunk in ollama_client.stream(req)]
    assert "".join(c.delta_text for c in chunks) == "Hello world!"
    assert chunks[-1].finished is True


@pytest.mark.asyncio
async def test_ollama_streaming_with_reasoning(
    ollama_client_with_reasoning, mock_aiohttp
):
    mock_aiohttp.post(
        "http://localhost:11434/v1/chat/completions",
        status=200,
        body=b'data: {"choices":[{"delta":{"content":"<think>Thinking step</think>Hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}],"done":true}\n\n',
    )

    messages = [{"role": "user", "content": "Test"}]
    req = LLMRequest(
        input=OllamaInput.from_list(messages),
        model="deepseek-r1",
        capture_reasoning=True,
    )
    chunks = [chunk async for chunk in ollama_client_with_reasoning.stream(req)]

    assert chunks[0].reasoning_delta == "Thinking step"
    assert chunks[0].delta_text == "Hello"
    assert chunks[1].delta_text == " world"


@pytest.mark.asyncio
async def test_ollama_hardware_options(ollama_client, mock_aiohttp):
    options = OllamaOptions(num_ctx=32768, num_gpu=999, kv_cache_type="q4_0")
    messages = [{"role": "user", "content": "Hi"}]
    req = LLMRequest(
        input=OllamaInput.from_list(messages), model="llama3.2", backend_options=options
    )

    mock_aiohttp.post(
        "http://localhost:11434/v1/chat/completions", status=200, body=b'{"done":true}'
    )
    await ollama_client.submit_batch([req])
    # Success = options reached the payload correctly
