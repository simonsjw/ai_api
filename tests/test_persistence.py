"""
Persistence + reasoning tests — fully mocked.
Uses isolated client + payload= for reliable JSON responses.
"""

import json

import pytest
from aioresponses import aioresponses

from ai_api.clients.grok.client import GrokConcreteClient
from ai_api.core.request import LLMRequest
from ai_api.data_structures.LLM_types_grok import GrokInput


@pytest.fixture
async def grok_persistence_client():
    """Local client for persistence tests only — isolated from the shared fixture."""
    client = GrokConcreteClient(
        model="grok-2",
        api_key="dummy-test-key-for-persistence-tests",
    )
    yield client
    if hasattr(client, "session") and client.session is not None:
        await client.session.close()


@pytest.mark.asyncio
async def test_json_files_persistence(grok_persistence_client, tmp_path, monkeypatch):
    """JSON files mode should write the raw response."""
    client = grok_persistence_client
    monkeypatch.setattr(client, "save_mode", "json_files")
    monkeypatch.setattr(client, "output_dir", tmp_path)

    with aioresponses() as m:
        m.post(
            "https://api.x.ai/v1/responses",
            status=200,
            payload={                                                                     # ← THIS IS THE FIX
                "id": "resp_123",
                "choices": [{"message": {"content": "Hello from Grok!"}}],
                "model": "grok-2",
            },
        )

        req = LLMRequest(
            input=GrokInput.from_list([{"role": "user", "content": "Test"}]),
            model="grok-2",
        )
        await client.submit_batch([req])

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        content = json.loads(files[0].read_text())
        assert content["choices"][0]["message"]["content"] == "Hello from Grok!"


@pytest.mark.asyncio
async def test_postgres_persistence_called(grok_persistence_client, mocker):
    """Postgres mode should call the save methods."""
    client = grok_persistence_client
    client.save_mode = "postgres"
    client.pg_settings = {}

    mock_save_req = mocker.patch.object(
        client, "_save_request_to_postgres", return_value=(None, None)
    )
    mock_save_resp = mocker.patch.object(client, "_save_response_to_postgres")
    mocker.patch.object(client, "_get_pool")

    with aioresponses() as m:
        m.post(
            "https://api.x.ai/v1/responses",
            status=200,
            payload={
                "id": "resp_123",
                "choices": [{"message": {"content": "Hello from Grok!"}}],
                "model": "grok-2",
            },
        )

        req = LLMRequest(
            input=GrokInput.from_list([{"role": "user", "content": "Test"}]),
            model="grok-2",
        )
        await client.submit_batch([req])

        mock_save_req.assert_called_once()
        mock_save_resp.assert_called_once()


@pytest.mark.asyncio
async def test_json_files_with_reasoning(
    grok_persistence_client, tmp_path, monkeypatch
):
    """capture_reasoning=True still works with JSON persistence."""
    client = grok_persistence_client
    monkeypatch.setattr(client, "save_mode", "json_files")
    monkeypatch.setattr(client, "output_dir", tmp_path)

    with aioresponses() as m:
        m.post(
            "https://api.x.ai/v1/responses",
            status=200,
            payload={
                "id": "resp_reasoning_123",
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "Thought process",
                        }
                    }
                ],
                "model": "grok-2",
            },
        )

        req = LLMRequest(
            input=GrokInput.from_list([{"role": "user", "content": "Test"}]),
            model="grok-2",
            capture_reasoning=True,
        )
        responses = await client.submit_batch([req])

        assert len(responses) == 1
        assert responses[0].reasoning_content == "Thought process"
        assert responses[0].content == "Answer"

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        saved_data = json.loads(files[0].read_text())
        msg = saved_data["choices"][0]["message"]
        assert msg["content"] == "Answer"
        assert msg["reasoning_content"] == "Thought process"


@pytest.mark.asyncio
async def test_no_reasoning_when_flag_false(grok_persistence_client):
    """Default capture_reasoning=False leaves reasoning as None."""
    client = grok_persistence_client
    with aioresponses() as m:
        m.post(
            "https://api.x.ai/v1/responses",
            status=200,
            payload={
                "id": "resp_no_reasoning_123",
                "choices": [
                    {"message": {"content": "<think>Hidden</think>Visible answer only"}}
                ],
                "model": "grok-2",
            },
        )

        req = LLMRequest(
            input=GrokInput.from_list([{"role": "user", "content": "Test"}]),
            model="grok-2",
        )
        responses = await client.submit_batch([req])

        assert len(responses) == 1
        assert responses[0].reasoning_content is None
        assert responses[0].content == "<think>Hidden</think>Visible answer only"


@pytest.mark.asyncio
async def test_postgres_saves_reasoning(grok_persistence_client, mocker):
    """Postgres mode works with capture_reasoning=True."""
    client = grok_persistence_client
    client.save_mode = "postgres"
    client.pg_settings = {}

    mock_save_req = mocker.patch.object(
        client, "_save_request_to_postgres", return_value=(None, None)
    )
    mock_save_resp = mocker.patch.object(client, "_save_response_to_postgres")
    mocker.patch.object(client, "_get_pool")

    with aioresponses() as m:
        m.post(
            "https://api.x.ai/v1/responses",
            status=200,
            payload={
                "id": "resp_reasoning_pg_123",
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "Thought",
                        }
                    }
                ],
                "model": "grok-2",
            },
        )

        req = LLMRequest(
            input=GrokInput.from_list([{"role": "user", "content": "Test"}]),
            model="grok-2",
            capture_reasoning=True,
        )
        await client.submit_batch([req])

        mock_save_req.assert_called_once()
        mock_save_resp.assert_called_once()
