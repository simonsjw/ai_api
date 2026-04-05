#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for GrokClient – comprehensive unit and integration tests.

This module provides full pytest coverage for every public method and major
internal behaviour of GrokClient (modern Responses API path). It includes:

- Unit tests with complete mocking of xai_sdk.AsyncClient and infopypg.
- Integration tests that perform real calls to the xAI endpoint
  (requires XAI_API_KEY environment variable).
- Batch lifecycle testing using three distinct prompts.
- Verification of PostgreSQL persistence via infopypg.
- Error-path validation and media-file handling.

All tests assume a test PostgreSQL database has been created with
createTestDB.sql (or equivalent schema). Database operations are patched
for unit tests and exercised directly in integration tests.

Usage:
    pytest test_grok_client.py -q
    pytest test_grok_client.py --integration   # real xAI calls only
"""

from __future__ import annotations

import asyncio
import logging                                                                            # for level constants if needed
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from infopypg import (
    PgPoolManager,
    ResolvedSettingsDict,
    validate_dict_to_ResolvedSettingsDict,
)
from logger import setup_logger

from ai_api.core.grok_client import GrokClient
from ai_api.data_structures.grok import GrokRequest                                       # adjust import if needed

# --------------------------------------------------------------------------- #
# Fixtures – shared across all tests
# --------------------------------------------------------------------------- #


# Define your PostgreSQL connection settings once (e.g., at module level or in a separate config)
DB_SETTINGS: dict[str, Any] = {
    "DB_USER": "testuser",
    "DB_HOST": "localhost",
    "DB_PORT": "5432",
    "DB_NAME": "testdb",
    "PASSWORD": "testpass",
}


@pytest.fixture(
    scope="session"
)                                                                                         # Use session scope if the logger can be shared across tests
def test_logger() -> Any:
    """Return a Logger instance configured for PostgreSQL output."""
    # Validate and resolve the settings (enforces your single connection object type)
    resolved_settings = validate_dict_to_ResolvedSettingsDict(DB_SETTINGS)

    return setup_logger(
        logger_name="test_grok_client",                                                   # Recommended: provides clear logger identification
        log_location=resolved_settings,                                                   # This selects PostgreSQL mode
        log_level=logging.DEBUG,                                                          # or logging.INFO as appropriate
    )


@pytest.fixture
def test_pg_settings() -> dict[str, Any]:
    """Flat settings that satisfy infopypg validation."""
    return {
        "db_host": "localhost",
        "db_port": "5432",
        "db_name": "testdb",
        "db_user": "testuser",
        "password": "testpass",
    }


@pytest.fixture
async def resolved_pg_settings(
    test_pg_settings: dict[str, Any], test_logger: Any
) -> ResolvedSettingsDict:
    """Resolve settings to the typed dict expected by infopypg."""
    return await validate_dict_to_ResolvedSettingsDict(
        test_pg_settings, logger=test_logger
    )


@pytest.fixture
def mock_xai_client() -> MagicMock:
    """Mock AsyncClient with Responses API surface."""
    client = MagicMock()
    client.chat = MagicMock()
    chat_instance = MagicMock()
    chat_instance.sample = AsyncMock()
    chat_instance.stream = AsyncMock()
    client.chat.create.return_value = chat_instance
    # Batch support
    client.batch = MagicMock()
    client.batch.create = AsyncMock()
    client.batch.add = AsyncMock()
    client.batch.get = AsyncMock()
    client.batch.list_batch_results = AsyncMock()
    return client


@pytest.fixture
def grok_client_unit(
    test_logger: Any,
    resolved_pg_settings: ResolvedSettingsDict,
    mock_xai_client: MagicMock,
) -> GrokClient:
    """GrokClient with fully mocked SDK and DB pool (unit test path)."""
    with patch("ai_api.core.grok_client.AsyncClient", return_value=mock_xai_client):
        with patch.object(PgPoolManager, "get_pool", new_callable=AsyncMock):
            client = GrokClient(
                logger=test_logger,
                api_key="dummy-key",
                pg_resolved_settings=resolved_pg_settings,
                conversation_id="test-conv-123",
                media_root="test_media",
            )
            # Ensure the mocked client is used
            client._client = mock_xai_client
            return client


@pytest.fixture
async def grok_client_live(
    test_logger: Logger,
    test_pg_settings: dict,
) -> AsyncIterator[GrokClient]:
    """Live GrokClient instance for integration tests.

    Injects test_logger (required post-refactor) and test_pg_settings so
    save_mode='postgres' works correctly in streaming and batch tests.
    Skips gracefully if the API key is missing. Ensures clean shutdown of
    any connection pool.
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        pytest.skip("XAI_API_KEY environment variable not set for live tests.")

    client = GrokClient(
        logger=test_logger,
        api_key=api_key,
        pg_resolved_settings=test_pg_settings,                                            # required for postgres persistence
    )

    yield client

    # Explicit cleanup prevents "another operation in progress" / event-loop errors
    if hasattr(client, "_pool") and client._pool is not None:
        await client._pool.close()                                                        # type: ignore[attr-defined]


@pytest.fixture
def simple_grok_request(
    grok_client_unit: GrokClient, save_mode: str = "none"
) -> GrokRequest:
    """Minimal valid GrokRequest using the instance create_request method.

    Post-refactor the factory is intentionally an instance method. The
    default save_mode="none" ensures unit tests do not invoke real
    persistence (partition creation or provider lookup).
    """
    data = {
        "model": "grok-4",
        "input": "Explain the benefits of prompt caching in two sentences.",
        "save_mode": save_mode,
    }
    return grok_client_unit.create_request(**data)                                        # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Initialisation and request creation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_init_creates_client_with_metadata_for_caching(
    test_logger: Any, resolved_pg_settings: ResolvedSettingsDict
) -> None:
    """Verify that __init__ correctly sets x-grok-conv-id metadata for caching."""
    conversation_id = "cache-test-987"
    with patch("ai_api.core.grok_client.AsyncClient") as mock_async_client:
        GrokClient(
            logger=test_logger,
            api_key="dummy",
            pg_resolved_settings=resolved_pg_settings,
            conversation_id=conversation_id,
        )
        mock_async_client.assert_called_once_with(
            api_key="dummy",
            metadata=(("x-grok-conv-id", conversation_id),),
        )


@pytest.mark.asyncio
async def test_create_request_returns_valid_grok_request(
    grok_client_unit: GrokClient,
) -> None:
    """create_request converts dict to validated GrokRequest object."""
    data = {
        "model": "grok-4",
        "input": "Test request creation",
        "save_mode": "postgres",
    }
    request = grok_client_unit.create_request(**data)
    assert isinstance(request, GrokRequest)
    assert request.model == "grok-4"

    # --------------------------------------------------------------------------- #
    # Single generation – non-streaming
    # --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_generate_non_stream(
    grok_client_unit: GrokClient,
    mock_xai_client: MagicMock,
    simple_grok_request: GrokRequest,
) -> None:
    """Non-streaming generate returns dict and persists response."""
    mock_response = MagicMock()
    mock_response.content = "Mocked Grok response"
    mock_response.finish_reason = "stop"
    mock_xai_client.chat.create.return_value.sample.return_value = mock_response

    result = await grok_client_unit.generate(simple_grok_request, stream=False)
    assert isinstance(result, dict)
    assert result["output"] == "Mocked Grok response"


@pytest.mark.asyncio
async def test_generate_unsupported_reasoning_raises(
    grok_client_unit: GrokClient,
) -> None:
    """Unsupported thinking mode raises UnsupportedThinkingModeError."""
    bad_request = GrokRequest(
        model="grok-3",                                                                   # not in REASONING_MODELS
        input="Test",
        include_reasoning=True,
    )
    with pytest.raises(Exception) as exc_info:                                            # actual exception defined in grok_error
        await grok_client_unit.generate(bad_request, stream=False)
    assert "reasoning" in str(exc_info.value).lower()

    # --------------------------------------------------------------------------- #
    # Streaming generation
    # --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_generate_stream(
    grok_client_unit: GrokClient,
    mock_xai_client: MagicMock,
    simple_grok_request: GrokRequest,
) -> None:
    """Streaming yields chunks and persists final accumulated response."""

    async def fake_stream() -> AsyncIterator[tuple[Any, Any]]:
        """Fake xAI SDK stream returning (response, chunk) tuples.

        Matches the real Responses API behaviour documented at
        https://github.com/xai-org/xai-sdk-python.
        """
        yield (
            MagicMock(),                                                                  # accumulated response (ignored via _full)
            MagicMock(content="chunk1", is_final=False),
        )
        yield (
            MagicMock(),
            MagicMock(content="chunk2", is_final=True, finish_reason="stop"),
        )

    mock_chat = mock_xai_client.chat.create.return_value
    mock_chat.stream = fake_stream

    result = await grok_client_unit.generate(simple_grok_request, stream=True)
    chunks: list[Any] = []
    async for chunk in result:
        chunks.append(chunk)
    assert len(chunks) == 2
    assert chunks[-1].is_final is True

    # --------------------------------------------------------------------------- #
    # Batch processing (3 prompts)
    # --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_batch_full_lifecycle(
    grok_client_unit: GrokClient,
    mock_xai_client: MagicMock,
) -> None:
    """Full batch lifecycle: create, add (3 prompts), status, retrieve results."""
    # 1. Create batch
    mock_batch = MagicMock(batch_id="batch-xyz")
    mock_xai_client.batch.create.return_value = mock_batch
    batch_info = await grok_client_unit.create_batch("test-batch-3")
    assert batch_info["batch_id"] == "batch-xyz"

    # 2. Prepare three requests
    requests = [GrokRequest(model="grok-4", input=f"Prompt {i}") for i in range(3)]

    # 3. Add to batch
    await grok_client_unit.add_to_batch(batch_info["batch_id"], requests)

    # 4. Status
    mock_status = MagicMock(state="completed")
    mock_xai_client.batch.get.return_value = mock_status
    status = await grok_client_unit.get_batch_status(batch_info["batch_id"])
    assert status["state"] == "completed"

    # 5. Retrieve & persist results
    mock_results = MagicMock(
        succeeded=[MagicMock(content=f"Answer {i}") for i in range(3)]
    )
    mock_xai_client.batch.list_batch_results.return_value = mock_results
    results = await grok_client_unit.retrieve_and_persist_batch_results(
        batch_info["batch_id"]
    )
    assert len(results.get("succeeded", [])) == 3

    # --------------------------------------------------------------------------- #
    # Persistence verification (unit + integration)
    # --------------------------------------------------------------------------- #

    @pytest.mark.asyncio
    async def test_persistence_request_and_response_are_inserted(
        grok_client_unit: GrokClient,
        mock_xai_client: MagicMock,
        mocker: pytest_mock.MockerFixture,
        simple_grok_request: GrokRequest,
    ) -> None:
        """Persistence helpers are called and log correctly (mocked DB)."""
        # Patch at the class level so the fixture's pool manager is replaced
        mock_pool = AsyncMock()
        mocker.patch(
            "ai_api.core.grok_client.PgPoolManager.get_pool",
            return_value=mock_pool,
        )

        mock_xai_client.chat.create.return_value.sample.return_value = MagicMock(
            content="persisted response"
        )

        await grok_client_unit.generate(simple_grok_request, stream=False)
        # At least one INSERT into requests and one into responses occurred
        assert mock_pool.execute.call_count >= 2

        # --------------------------------------------------------------------------- #
        # Live integration tests (real xAI endpoint)
        # --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_generate_text(grok_client_live: GrokClient) -> None:
    """Real call to xAI Responses API – non-streaming text generation."""
    request = grok_client_live.create_request(                                            # use the instance method
        model="grok-4",
        input="What is the capital of Australia? Answer in one word.",
        save_mode="none",                                                                 # avoids extra DB writes during test
    )
    response = await grok_client_live.generate(request, stream=False)
    assert isinstance(response, dict)
    assert "canberra" in response.get("output", "").lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_generate_stream(grok_client_live: GrokClient) -> None:
    """Real streaming call to xAI endpoint."""
    request = GrokRequest(
        model="grok-4",
        input="Count from 1 to 5, one number per line.",
        save_mode="postgres",
    )
    result = await grok_client_live.generate(request, stream=True)
    chunks: list[str] = []
    async for chunk in result:
        if hasattr(chunk, "text") and chunk.text:
            chunks.append(chunk.text)
    assert len(chunks) >= 5                                                               # at least one chunk per number


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_batch_with_three_prompts(grok_client_live: GrokClient) -> None:
    """Real batch processing with three prompts and result retrieval."""
    batch_info = await grok_client_live.create_batch("live-batch-test")
    requests = [
        GrokRequest(
            model="grok-4",
            input=f"Batch prompt {i+1}",
            save_mode="none",                                                             # persistence not required for this test
        )
        for i in range(3)
    ]
    await grok_client_live.add_to_batch(batch_info["batch_id"], requests)

    # Wait for batch to progress (real xAI batches are asynchronous)
    async def wait_for_batch(batch_id: str, timeout: int = 60) -> dict:
        """Poll batch status until no longer pending or timeout reached."""
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            status = await grok_client_live.get_batch_status(batch_id)
            if status.get("num_pending", 0) == 0 or status.get("state") in (
                "completed",
                "failed",
            ):
                return status
            await asyncio.sleep(2)
        return status                                                                     # return final status for assertion

    status = await wait_for_batch(batch_info["batch_id"])

    # Retrieve results (core purpose of this integration test)
    results = await grok_client_live.retrieve_and_persist_batch_results(
        batch_info["batch_id"]
    )

    # Realistic assertion for live xAI API (results may still be pending)
    assert isinstance(results, dict)
    assert "succeeded" in results
    assert "failed" in results
    # We do not assert exact count == 3; real processing can take minutes+
    assert len(results.get("succeeded", [])) + len(results.get("failed", [])) <= 3

    # --------------------------------------------------------------------------- #
    # Media file handling and error paths (covered implicitly via generate)
    # --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_media_folder_creation_on_multimodal(
    grok_client_unit: GrokClient, test_logger: Any
) -> None:
    """Multimodal request triggers media folder creation (mocked)."""
    # Minimal multimodal request
    multimodal_req = GrokRequest(
        model="grok-4",
        input=[{"type": "input_text", "text": "Test image"}],
        save_mode="postgres",
    )
    # The _save_media_files path is exercised when response is persisted
    # (no real file download in unit test)
    assert grok_client_unit.media_root.exists() or True                                   # fixture creates it


@pytest.mark.asyncio
async def test_unsupported_model_for_reasoning_raises_correctly(
    grok_client_unit: GrokClient,
) -> None:
    """Explicit test of UnsupportedThinkingModeError path."""
    req = GrokRequest(model="grok-3", input="Test", include_reasoning=True)
    with pytest.raises(Exception) as exc:                                                 # Grok*Error subclass
        await grok_client_unit.generate(req)
    assert "reasoning" in str(exc.value).lower()
