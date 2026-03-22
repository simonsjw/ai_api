"""
Grok client tests — batch, streaming, conversation caching, and full reasoning support.
"""

import json
import re

import pytest

from ai_api.core.request import LLMRequest
from ai_api.data_structures.LLM_types_grok import GrokInput


@pytest.mark.asyncio
async def test_grok_batch_success(grok_client, mock_aiohttp):
    """Basic successful batch (no reasoning)."""
    mock_aiohttp.post(
        re.compile(r"https://api\.x\.ai/v1/responses.*"),
        status=200,
        body=json.dumps(
            {
                "id": "resp_123",
                "object": "chat.completion",
                "created_at": 1730000000,
                "model": "grok-2",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Hello from Grok!"}
                        ],
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25,
                },
            }
        ),
        repeat=True,
    )

    messages = [{"role": "user", "content": "Hi!"}]
    req = LLMRequest(input=GrokInput.from_list(messages), model="grok-2")
    responses = await grok_client.submit_batch([req])

    assert len(responses) == 1
    assert responses[0].text == "Hello from Grok!"
    assert responses[0].reasoning_content is None


@pytest.mark.asyncio
async def test_grok_batch_with_reasoning(grok_client, mock_aiohttp):
    """Batch mode with reasoning extraction."""
    mock_aiohttp.post(
        re.compile(r"https://api\.x\.ai/v1/responses.*"),
        status=200,
        body=json.dumps(
            {
                "id": "resp_456",
                "object": "chat.completion",
                "created_at": 1730000001,
                "model": "grok-2",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Final answer"}],
                    }
                ],
                "choices": [                                                              # kept for reasoning_content extraction
                    {"message": {"reasoning_content": "Step 1: 2+2=4"}}
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 18,
                    "total_tokens": 30,
                },
            }
        ),
        repeat=True,
    )

    req = LLMRequest(
        input=GrokInput.from_list([{"role": "user", "content": "2+2?"}]),
        model="grok-2",
        capture_reasoning=True,
    )
    resp = (await grok_client.submit_batch([req]))[0]

    assert resp.text == "Final answer"
    assert resp.reasoning_content == "Step 1: 2+2=4"


@pytest.mark.asyncio
async def test_grok_streaming_with_reasoning(grok_client, mock_aiohttp):
    """Live reasoning deltas in streaming."""
    mock_aiohttp.post(
        "https://api.x.ai/v1/responses",
        status=200,
        body=b'data: {"choices":[{"delta":{"content":"Final answer","reasoning_content":"Thinking step"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"!"}}],"done":true}\n\n',
        repeat=True,
    )

    req = LLMRequest(
        input=GrokInput.from_list([{"role": "user", "content": "Test"}]),
        model="grok-2",
        capture_reasoning=True,
    )
    chunks = [chunk async for chunk in grok_client.stream(req)]

    assert len(chunks) == 2
    assert chunks[0].delta_text == "Final answer"
    assert chunks[0].reasoning_delta == "Thinking step"
    assert chunks[1].delta_text == "!"
    assert chunks[1].finished is True


@pytest.mark.asyncio
async def test_conv_id_caching(grok_client):
    """Original legacy conversation caching behaviour."""
    grok_client._set_conv_id = True
    first_id = grok_client.conv_id
    assert first_id is not None
    assert grok_client.conv_id == first_id                                                # same ID reused
