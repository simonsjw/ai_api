import json

import pytest

from ai_api.core.request import LLMRequest
from ai_api.data_structures.LLM_types_grok import GrokInput


@pytest.mark.asyncio
async def test_submit_batch_success(grok_client, mock_aiohttp):
    # Simulate successful Grok API response
    mock_aiohttp.post(
        "https://api.x.ai/v1/responses",
        status=200,
        body=json.dumps(
            {
                "id": "resp_123",
                "choices": [{"message": {"content": "Hello from Grok!"}}],
                "model": "grok-2",
            }
        ),
        headers={"content-type": "application/json"},
    )

    messages = [{"role": "user", "content": "Hi!"}]
    req = LLMRequest(input=GrokInput.from_list(messages), model="grok-2")

    responses = await grok_client.submit_batch([req])
    assert len(responses) == 1
    assert responses[0].content == "Hello from Grok!"
    assert responses[0].model == "grok-2"


@pytest.mark.asyncio
async def test_conv_id_caching(grok_client):
    """Original legacy behaviour preserved"""
    grok_client._set_conv_id = True
    assert grok_client.conv_id is not None
    first_id = grok_client.conv_id
    assert grok_client.conv_id == first_id                                                # same ID on subsequent calls


@pytest.mark.asyncio
async def test_streaming(grok_client, mock_aiohttp):
    # Simulate SSE stream
    mock_aiohttp.post(
        "https://api.x.ai/v1/responses",
        status=200,
        body=b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        b"data: [DONE]\n",
        headers={"content-type": "text/event-stream"},
    )

    req = LLMRequest(...)                                                                 # same as above
    chunks = []
    async for chunk in grok_client.stream(req):
        chunks.append(chunk.delta_text)

    assert "".join(chunks) == "Hello world"
