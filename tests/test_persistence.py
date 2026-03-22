"""
Persistence tests (JSON + Postgres) — now includes reasoning capture.
"""

import json
from pathlib import Path

import pytest

from ai_api.core.request import LLMRequest
from ai_api.data_structures.LLM_types_grok import GrokInput                               # or OllamaInput


@pytest.mark.asyncio
async def test_json_files_with_reasoning(
    grok_client, tmp_path, monkeypatch, mock_aiohttp
):
    monkeypatch.setattr(grok_client, "save_mode", "json_files")
    monkeypatch.setattr(grok_client, "output_dir", tmp_path)

    mock_aiohttp.post(
        ...,
        status=200,
        body=json.dumps(
            {"choices": [{"message": {"content": "<think>Reasoned</think>Answer"}}]}
        ),
    )

    req = LLMRequest(input=..., model="grok-2", capture_reasoning=True)
    await grok_client.submit_batch([req])

    saved = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
    assert "reasoning_content" in saved["choices"][0]["message"]


@pytest.mark.asyncio
async def test_postgres_saves_reasoning(grok_client, mocker, mock_aiohttp):
    grok_client.save_mode = "postgres"
    mocker.patch.object(grok_client, "_save_request_to_postgres")
    mocker.patch.object(grok_client, "_save_response_to_postgres")

    mock_aiohttp.post(
        ...,
        status=200,
        body=json.dumps(
            {
                "choices": [
                    {"message": {"content": "Answer", "reasoning_content": "Thought"}}
                ]
            }
        ),
    )

    req = LLMRequest(..., capture_reasoning=True)
    resp = (await grok_client.submit_batch([req]))[0]

    assert resp.reasoning_content == "Thought"


@pytest.mark.asyncio
async def test_no_reasoning_when_flag_false(grok_client, mock_aiohttp):
    mock_aiohttp.post(
        ...,
        status=200,
        body=json.dumps(
            {"choices": [{"message": {"content": "<think>Hidden</think>Visible"}}]}
        ),
    )

    req = LLMRequest(..., capture_reasoning=False)
    resp = (await grok_client.submit_batch([req]))[0]

    assert resp.reasoning_content is None
    assert resp.text == "<think>Hidden</think>Visible"                                    # raw content unchanged
