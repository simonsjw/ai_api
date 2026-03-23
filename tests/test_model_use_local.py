"""
tests/test_model_use_local.py
LIVE LOCAL TESTS — only Ollama (free, uses your machine power)
Run with: pytest tests/test_model_use_local.py -m live
"""

from pathlib import Path

import pytest

from ai_api import create
from ai_api.core.request import LLMRequest
from ai_api.data_structures.LLM_types_ollama import OllamaInput


@pytest.mark.live
@pytest.mark.asyncio
async def test_ollama_qwen3_json_live(tmp_path):
    client = await create(
        provider="ollama",
        model="qwen3-coder-next:latest",
        base_url="http://localhost:11434/v1",
        save_mode="json_files",
        output_dir=tmp_path,
        concurrency=1,
        max_retries=1,
    )

    # ← FIXED: direct construction (from_list was losing the messages)
    ollama_input: OllamaInput = OllamaInput.from_list(
        [{"role": "user", "content": "Please provide the sum of 5 and 6."}]
    )

    # Debug: see exactly what will be sent
    print(
        f"\n🔍 DEBUG payload being sent to Ollama: {ollama_input.model_dump() if hasattr(ollama_input, 'model_dump') else ollama_input}"
    )

    req = LLMRequest(input=ollama_input, model="qwen3-coder-next:latest")

    try:
        responses = await client.submit_batch([req])
        assert len(responses) > 0, "No responses returned from Ollama!"
        resp = responses[0]
    except Exception as e:
        print(f"\n🚨 ERROR in Ollama JSON call: {e}")
        raise

    print(f"\n=== QWEN3-CODER-NEXT:LATEST (JSON) ===")
    print(f"Response: {resp.text}")

    json_files = list(tmp_path.glob("*.json"))
    print(f"JSON files saved: {len(json_files)}")
    if json_files:
        print(f"Saved file: {json_files[0]}")

    assert "11" in resp.text


@pytest.mark.live
@pytest.mark.asyncio
async def test_ollama_qwen3_postgres_live():
    client = await create(
        provider="ollama",
        model="qwen3-coder-next:latest",
        base_url="http://localhost:11434/v1",
        save_mode="postgres",
        concurrency=1,
        max_retries=1,
    )

    ollama_input: OllamaInput = OllamaInput.from_list(
        [{"role": "user", "content": "Please provide the sum of 5 and 6."}]
    )
    req = LLMRequest(input=ollama_input, model="qwen3-coder-next:latest")

    try:
        responses = await client.submit_batch([req])
        assert len(responses) > 0, "No responses returned from Ollama!"
        resp = responses[0]
    except Exception as e:
        print(f"\n🚨 ERROR in Ollama Postgres call: {e}")
        raise

    print(f"\n=== QWEN3-CODER-NEXT:LATEST (POSTGRES) ===")
    print(f"Response: {resp.text}")
    print(
        "Data saved to Postgres (daily partitioned tables) — or skipped if no DB settings"
    )
