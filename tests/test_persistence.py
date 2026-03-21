"""
Persistence mode tests (JSON files + PostgreSQL).
Uses pytest-mock to verify the exact save paths without touching real disk or DB.
"""

import json
from pathlib import Path

import pytest

from ai_api.core.request import LLMRequest
from ai_api.data_structures.LLM_types_grok import GrokInput                               # or OllamaInput


@pytest.mark.asyncio
async def test_json_files_persistence(grok_client, tmp_path, monkeypatch):
    """Verify JSON file is written when save_mode="json_files"."""
    monkeypatch.setattr(grok_client, "save_mode", "json_files")
    monkeypatch.setattr(grok_client, "output_dir", tmp_path)

    mock_aiohttp.post(...)                                                                # (same successful response as in test_grok_client.py)

    req = LLMRequest(
        input=GrokInput.from_list([{"role": "user", "content": "Test"}]), model="grok-2"
    )
    await grok_client.submit_batch([req])

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    content = json.loads(files[0].read_text())
    assert "choices" in content


@pytest.mark.asyncio
async def test_postgres_persistence_called(grok_client, mocker):
    """Verify _save_request_to_postgres and _save_response_to_postgres are called when save_mode="postgres"."""
    grok_client.save_mode = "postgres"
    grok_client.pg_settings = {}                                                          # dummy

    # Mock the heavy DB methods
    mock_save_req = mocker.patch.object(
        grok_client, "_save_request_to_postgres", return_value=(None, None)
    )
    mock_save_resp = mocker.patch.object(grok_client, "_save_response_to_postgres")
    mock_get_pool = mocker.patch.object(grok_client, "_get_pool")

    # Mock successful HTTP
    mock_aiohttp.post(...)                                                                # same as before

    req = LLMRequest(...)                                                                 # same as above
    await grok_client.submit_batch([req])

    mock_save_req.assert_called_once()
    mock_save_resp.assert_called_once()
    mock_get_pool.assert_called()


@pytest.mark.asyncio
async def test_none_mode_no_saving(grok_client, mocker):
    """save_mode='none' should never call any persistence methods."""
    grok_client.save_mode = "none"

    mock_save_req = mocker.patch.object(grok_client, "_save_request_to_postgres")
    mock_save_resp = mocker.patch.object(grok_client, "_save_response_to_postgres")
    mock_file_write = mocker.patch("pathlib.Path.write_text")

    mock_aiohttp.post(...)                                                                # success

    req = LLMRequest(...)
    await grok_client.submit_batch([req])

    mock_save_req.assert_not_called()
    mock_save_resp.assert_not_called()
    mock_file_write.assert_not_called()


@pytest.mark.asyncio
async def test_return_responses_false_postgres(grok_client, mocker):
    """return_responses=False should still save to DB but return None."""
    grok_client.save_mode = "postgres"
    mocker.patch.object(grok_client, "_save_request_to_postgres")
    mocker.patch.object(grok_client, "_save_response_to_postgres")

    mock_aiohttp.post(...)                                                                # success

    req = LLMRequest(...)
    result = await grok_client.submit_batch([req], return_responses=False)

    assert result is None
