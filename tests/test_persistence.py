"""
Persistence tests (JSON + Postgres) — now includes reasoning capture.
"""

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from aioresponses import aioresponses                                                     # ← Add this import at the top of the file

from ai_api.core.request import LLMRequest
from ai_api.data_structures.LLM_types_grok import GrokInput


@pytest.mark.asyncio
async def test_json_files_persistence(grok_client, tmp_path, monkeypatch, mock_aiohttp):
    """JSON files persistence — uses the repo's official mocking pattern."""
    # 1. Switch client to json_files mode (fixture defaults to "none")
    monkeypatch.setattr(grok_client, "save_mode", "json_files")
    monkeypatch.setattr(grok_client, "output_dir", tmp_path)

    # 2. Register the mock using the fixture (this is the CORRECT way)
    mock_aiohttp.post(
        re.compile(r"https://api\.x\.ai/v1/responses.*"),
        status=200,
        payload={
            "id": "resp_test_123456",
            "created_at": 1743000000,                                                     # ← REQUIRED (Unix timestamp)
            "model": "grok-2",                                                            # ← REQUIRED
            "object": "response",
            "output": [                                                                   # ← native Grok format (used by GrokResponse.text)
                {
                    "id": "msg_test_001",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "The new result is 3030."}
                    ],
                    "status": "completed",
                }
            ],
            "choices": [                                                                  # ← still needed for reasoning_content extraction
                {
                    "message": {
                        "reasoning_content": "First, the user asked for 101 * 3.\nStep 1: 100 * 3 = 300..."
                    }
                }
            ],
            "usage": {"prompt_tokens": 45, "completion_tokens": 12, "total_tokens": 57},
        },
        repeat=True,
    )

    # 3. Build request with the exact types the library expects
    messages = [{"role": "user", "content": "hello"}]
    input_obj = GrokInput.from_list(messages)
    req = LLMRequest(input=input_obj, model="grok-2")

    # 4. This now hits the mock — no real network call
    await grok_client.submit_batch([req])

    # 5. Verify the file was written (this will now pass)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1, f"Expected 1 file, found {len(files)}"

    saved_content = files[0].read_text()
    assert "The new result is 3030." in saved_content
    assert "reasoning_content" in saved_content


@pytest.mark.asyncio
async def test_postgres_persistence_called(grok_client, mocker, mock_aiohttp):
    """Postgres mode should call the save functions (no real DB needed)."""
    grok_client.save_mode = "postgres"
    mock_save_req = mocker.patch.object(grok_client, "_save_request_to_postgres")
    mock_save_resp = mocker.patch.object(grok_client, "_save_response_to_postgres")
    mock_save_req.return_value = (
        uuid.uuid4(),
        datetime.now(UTC),
    )                                                                                     # prevents unpack error

    mock_aiohttp.post(
        re.compile(r"https://api\.x\.ai/v1/responses.*"),
        status=200,
        payload={
            "id": "resp_test_123456",
            "created_at": 1743000000,
            "model": "grok-2",
            "object": "response",
            "output": [
                {
                    "id": "msg_test_001",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "The new result is 3030."}
                    ],
                    "status": "completed",
                }
            ],
            "choices": [
                {
                    "message": {
                        "reasoning_content": "First, the user asked for 101 * 3.\nStep 1: 100 * 3 = 300..."
                    }
                }
            ],
            "usage": {"prompt_tokens": 45, "completion_tokens": 12, "total_tokens": 57},
        },
        repeat=True,
    )

    messages = [{"role": "user", "content": "hello"}]
    input_obj = GrokInput.from_list(messages)
    req = LLMRequest(input=input_obj, model="grok-2")

    await grok_client.submit_batch([req])

    mock_save_resp.assert_called_once()


@pytest.mark.asyncio
async def test_json_files_with_reasoning(
    grok_client, tmp_path, monkeypatch, mock_aiohttp
):
    """JSON files mode with capture_reasoning=True should save reasoning_content."""
    monkeypatch.setattr(grok_client, "save_mode", "json_files")
    monkeypatch.setattr(grok_client, "output_dir", tmp_path)

    mock_aiohttp.post(
        re.compile(r"https://api\.x\.ai/v1/responses.*"),
        status=200,
        payload={
            "id": "resp_test_123456",
            "created_at": 1743000000,
            "model": "grok-2",
            "object": "response",
            "output": [
                {
                    "id": "msg_test_001",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "The new result is 3030."}
                    ],
                    "status": "completed",
                }
            ],
            "choices": [
                {
                    "message": {
                        "reasoning_content": "First, the user asked for 101 * 3.\nStep 1: 100 * 3 = 300..."
                    }
                }
            ],
            "usage": {"prompt_tokens": 45, "completion_tokens": 12, "total_tokens": 57},
        },
        repeat=True,
    )

    messages = [{"role": "user", "content": "hello"}]
    input_obj = GrokInput.from_list(messages)
    req = LLMRequest(input=input_obj, model="grok-2", capture_reasoning=True)

    await grok_client.submit_batch([req])

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    saved_content = files[0].read_text()
    assert "The new result is 3030." in saved_content
    assert "reasoning_content" in saved_content


@pytest.mark.asyncio
async def test_no_reasoning_when_flag_false(grok_client, mock_aiohttp):
    """When capture_reasoning=False, reasoning_content should be None and text extracted correctly."""
    mock_aiohttp.post(
        re.compile(r"https://api\.x\.ai/v1/responses.*"),
        status=200,
        payload={
            "id": "resp_test_123456",
            "created_at": 1743000000,
            "model": "grok-2",
            "object": "response",
            "output": [
                {
                    "type": "message",                                                    # ← THIS WAS MISSING
                    "id": "msg_test_001",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "The new result is 3030."}
                    ],
                    "status": "completed",
                }
            ],
            "choices": [                                                                  # kept for legacy compatibility
                {"message": {"role": "assistant", "content": "The new result is 3030."}}
            ],
            "usage": {"prompt_tokens": 45, "completion_tokens": 12, "total_tokens": 57},
        },
        repeat=True,
    )

    messages = [{"role": "user", "content": "hello"}]
    input_obj = GrokInput.from_list(messages)
    req = LLMRequest(input=input_obj, model="grok-2", capture_reasoning=False)

    resp = (await grok_client.submit_batch([req]))[0]

    assert resp.reasoning_content is None
    assert resp.text == "The new result is 3030."


@pytest.mark.asyncio
async def test_postgres_saves_reasoning(grok_client, mocker, mock_aiohttp):
    """Postgres mode with capture_reasoning=True should still call save_response."""
    grok_client.save_mode = "postgres"
    mock_save_req = mocker.patch.object(grok_client, "_save_request_to_postgres")
    mock_save_resp = mocker.patch.object(grok_client, "_save_response_to_postgres")
    mock_save_req.return_value = (uuid.uuid4(), datetime.now(UTC))

    mock_aiohttp.post(
        re.compile(r"https://api\.x\.ai/v1/responses.*"),
        status=200,
        payload={
            "id": "resp_test_123456",
            "created_at": 1743000000,
            "model": "grok-2",
            "object": "response",
            "output": [
                {
                    "id": "msg_test_001",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "The new result is 3030."}
                    ],
                    "status": "completed",
                }
            ],
            "choices": [
                {
                    "message": {
                        "reasoning_content": "First, the user asked for 101 * 3.\nStep 1: 100 * 3 = 300..."
                    }
                }
            ],
            "usage": {"prompt_tokens": 45, "completion_tokens": 12, "total_tokens": 57},
        },
        repeat=True,
    )

    messages = [{"role": "user", "content": "hello"}]
    input_obj = GrokInput.from_list(messages)
    req = LLMRequest(input=input_obj, model="grok-2", capture_reasoning=True)

    resp = (await grok_client.submit_batch([req]))[0]

    assert resp.reasoning_content is not None
    mock_save_resp.assert_called_once()
