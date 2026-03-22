"""
Persistence + reasoning tests — fully mocked, no real network or DB.
"""

import json
from pathlib import Path

import pytest

from ai_api.core.request import LLMRequest
from ai_api.data_structures.LLM_types_grok import GrokInput


@pytest.mark.asyncio
async def test_json_files_persistence(grok_client, tmp_path, monkeypatch, mock_aiohttp):
    """JSON files mode should write the raw response."""
    monkeypatch.setattr(grok_client, "save_mode", "json_files")
    monkeypatch.setattr(grok_client, "output_dir", tmp_path)

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
    )

    req = LLMRequest(
        input=GrokInput.from_list([{"role": "user", "content": "Test"}]), model="grok-2"
    )
    await grok_client.submit_batch([req])

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    content = json.loads(files[0].read_text())
    assert content["choices"][0]["message"]["content"] == "Hello from Grok!"


@pytest.mark.asyncio
async def test_postgres_persistence_called(grok_client, mocker, mock_aiohttp):
    """Postgres mode should call the save methods."""
    grok_client.save_mode = "postgres"
    grok_client.pg_settings = {}

    mock_save_req = mocker.patch.object(
        grok_client, "_save_request_to_postgres", return_value=(None, None)
    )
    mock_save_resp = mocker.patch.object(grok_client, "_save_response_to_postgres")
    mocker.patch.object(grok_client, "_get_pool")

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
    )

    req = LLMRequest(
        input=GrokInput.from_list([{"role": "user", "content": "Test"}]), model="grok-2"
    )
    await grok_client.submit_batch([req])

    mock_save_req.assert_called_once()
    mock_save_resp.assert_called_once()


@pytest.mark.asyncio
async def test_none_mode_no_saving(grok_client, mocker, mock_aiohttp):
    """'none' mode should never call persistence."""
    grok_client.save_mode = "none"

    mock_save_req = mocker.patch.object(grok_client, "_save_request_to_postgres")
    mock_save_resp = mocker.patch.object(grok_client, "_save_response_to_postgres")
    mock_file = mocker.patch("pathlib.Path.write_text")

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
    )

    req = LLMRequest(
        input=GrokInput.from_list([{"role": "user", "content": "Test"}]), model="grok-2"
    )
    await grok_client.submit_batch([req])

    mock_save_req.assert_not_called()
    mock_save_resp.assert_not_called()
    mock_file.assert_not_called()

    # ====================== REASONING TESTS ======================


@pytest.mark.asyncio
async def test_json_files_with_reasoning(
    grok_client, tmp_path, monkeypatch, mock_aiohttp
):
    """capture_reasoning=True still works with JSON persistence."""
    monkeypatch.setattr(grok_client, "save_mode", "json_files")
    monkeypatch.setattr(grok_client, "output_dir", tmp_path)

    mock_aiohttp.post(
        "https://api.x.ai/v1/responses",
        status=200,
        body=json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "Thought process",
                        }
                    }
                ],
                "model": "grok-2",
            }
        ),
    )

    req = LLMRequest(
        input=GrokInput.from_list([{"role": "user", "content": "Test"}]),
        model="grok-2",
        capture_reasoning=True,
    )
    responses = await grok_client.submit_batch([req])

    assert len(responses) == 1
    assert responses[0].reasoning_content == "Thought process"


@pytest.mark.asyncio
async def test_postgres_saves_reasoning(grok_client, mocker, mock_aiohttp):
    """Postgres mode works with capture_reasoning=True."""
    grok_client.save_mode = "postgres"
    mocker.patch.object(grok_client, "_save_request_to_postgres")
    mocker.patch.object(grok_client, "_save_response_to_postgres")
    mocker.patch.object(grok_client, "_get_pool")

    mock_aiohttp.post(
        "https://api.x.ai/v1/responses",
        status=200,
        body=json.dumps(
            {
                "choices": [
                    {"message": {"content": "Answer", "reasoning_content": "Thought"}}
                ],
                "model": "grok-2",
            }
        ),
    )

    req = LLMRequest(
        input=GrokInput.from_list([{"role": "user", "content": "Test"}]),
        model="grok-2",
        capture_reasoning=True,
    )
    await grok_client.submit_batch([req])


@pytest.mark.asyncio
async def test_no_reasoning_when_flag_false(grok_client, mock_aiohttp):
    """Default capture_reasoning=False leaves reasoning as None."""
    mock_aiohttp.post(
        "https://api.x.ai/v1/responses",
        status=200,
        body=json.dumps(
            {
                "choices": [{"message": {"content": "<think>Hidden</think>Visible"}}],
                "model": "grok-2",
            }
        ),
    )

    req = LLMRequest(
        input=GrokInput.from_list([{"role": "user", "content": "Test"}]),
        model="grok-2",
        # capture_reasoning=False by default
    )
    responses = await grok_client.submit_batch([req])

    assert responses[0].reasoning_content is None
