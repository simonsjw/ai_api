"""
tests/test_model_use_remote.py
LIVE REMOTE TESTS — only Grok (costs real money)
Run with: pytest tests/test_model_use_remote.py -m live
"""

import os
from pathlib import Path

import pytest

from ai_api import create
from ai_api.core.request import LLMRequest
from ai_api.data_structures.LLM_types_grok import GrokInput


@pytest.mark.live
@pytest.mark.asyncio
async def test_grok_4_1_fast_non_reasoning_json_live(tmp_path):
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        pytest.skip("Set XAI_API_KEY environment variable to run remote tests")

    client = await create(
        provider="grok",
        model="grok-4-1-fast-non-reasoning",
        api_key=api_key,
        save_mode="json_files",
        output_dir=tmp_path,
        concurrency=1,
        max_retries=1,
    )

    req = LLMRequest(
        input=GrokInput.from_list(
            [{"role": "user", "content": "Please provide the sum of 5 and 6."}]
        ),
        model="grok-4-1-fast-non-reasoning",
    )

    responses = await client.submit_batch([req])
    resp = responses[0]

    print(f"\n=== GROK-4-1-FAST-NON-REASONING (JSON) ===")
    print(f"Response: {resp.text}")
    print(f"Reasoning: {getattr(resp, 'reasoning_content', None)}")

    json_files = list(tmp_path.glob("*.json"))
    print(f"JSON files saved: {len(json_files)}")
    if json_files:
        print(f"Saved file: {json_files[0]}")

    assert "11" in resp.text


@pytest.mark.live
@pytest.mark.asyncio
async def test_grok_4_1_fast_non_reasoning_postgres_live():
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        pytest.skip("Set XAI_API_KEY environment variable to run remote tests")

    client = await create(
        provider="grok",
        model="grok-4-1-fast-non-reasoning",
        api_key=api_key,
        save_mode="postgres",
        concurrency=1,
        max_retries=1,
    )

    req = LLMRequest(
        input=GrokInput.from_list(
            [{"role": "user", "content": "Please provide the sum of 5 and 6."}]
        ),
        model="grok-4-1-fast-non-reasoning",
    )

    responses = await client.submit_batch([req])
    resp = responses[0]

    print(f"\n=== GROK-4-1-FAST-NON-REASONING (POSTGRES) ===")
    print(f"Response: {resp.text}")
    print("Data saved to Postgres (daily partitioned tables)")
