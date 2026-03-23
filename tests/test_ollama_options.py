"""
tests/test_ollama_options.py
UNIT + LIVE PERFORMANCE TESTS for the OllamaOptions module
Focus: quantization levels (kv_cache_type) and their real-world timing impact.

Run with:

# 1. Quick sanity check — no more warnings
    pytest tests/test_ollama_options.py -q --tb=no

# 2. See the benchmark table
    pytest tests/test_ollama_options.py -m live -q --tb=line --capture=no

"""

import time
from pathlib import Path

import pytest

from ai_api import create
from ai_api.clients.ollama.options import OllamaOptions                                   # ← the module under test
from ai_api.core.request import LLMRequest
from ai_api.data_structures.LLM_types_ollama import OllamaInput

# ========================== UNIT TESTS ==========================


def test_ollama_options_to_dict_filters_none():
    """OllamaOptions.to_dict() must exclude None values (clean payload)."""
    opts = OllamaOptions(
        num_gpu=999,
        num_thread=8,
        kv_cache_type="q4_0",
        keep_alive=None,                                                                  # should be dropped
    )
    d = opts.to_dict()

    assert d == {
        "num_gpu": 999,
        "num_thread": 8,
        "kv_cache_type": "q4_0",
    }
    assert "keep_alive" not in d


def test_ollama_options_empty_to_dict():
    """Empty options should produce empty dict."""
    assert OllamaOptions().to_dict() == {}

    # ========================== LIVE PERFORMANCE BENCHMARK ==========================


@pytest.mark.live
@pytest.mark.asyncio
async def test_ollama_quantization_performance(tmp_path):
    """Light diagnostic benchmark: short output + full error visibility."""
    client = await create(
        provider="ollama",
        model="qwen3-coder-next:latest",
        base_url="http://localhost:11434/v1",
        save_mode="json_files",
        output_dir=tmp_path,
        concurrency=1,
        max_retries=1,
    )

    prompt = "Explain quantum computing vs classical computing in 150 words."
    ollama_input = OllamaInput.from_list([{"role": "user", "content": prompt}])

    levels = [None, "f16", "q4_0"]                                                        # reduced to 3 levels (q8_0 often flaky)
    results = {}

    print("\n=== LIGHT DIAGNOSTIC KV-CACHE BENCHMARK ===")
    print(f"Model: qwen3-coder-next:latest   |   Max tokens: 150\n")

    for level in levels:
        opts = OllamaOptions(
            kv_cache_type=level,
            num_thread=8,
            num_gpu=-1,
        )

        req = LLMRequest(
            input=ollama_input,
            model="qwen3-coder-next:latest",
            backend_options=opts,
            max_output_tokens=150,                                                        # much lighter
            temperature=0.7,
        )

        print(f"\n→ Testing kv_cache_type={level or 'None'}...")

        # Minimal warm-up
        await client.submit_batch([req])

        # 3 timed runs with full error catching
        times = []
        for run in range(3):
            try:
                start = time.perf_counter()
                responses = await client.submit_batch([req])
                if not responses or len(responses) == 0:
                    raise RuntimeError(
                        f"submit_batch returned empty list on run {run+1}"
                    )
                resp = responses[0]
                end = time.perf_counter()

                token_count = len(resp.text.split())
                times.append(end - start)
                print(
                    f"  Run {run+1}/3: {end-start:.2f}s  (~{token_count} tokens) - OK"
                )
            except Exception as e:
                print(f"  ❌ CRASH on run {run+1} for {level or 'None'}:")
                print(f"     {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
                raise                                                                     # stop so you see everything

        avg_time = sum(times) / len(times)
        tps = 50 / avg_time if avg_time > 0 else 0                                        # rough TPS (prompt ~50 tokens)

        results[level or "None"] = round(tps, 1)
        print(f"kv_cache_type={level or 'None':6}  →  {tps:5.1f} TPS")

        # Summary
    print("\n=== SUMMARY TABLE (Tokens per Second) ===")
    print("kv_cache_type | TPS   | Relative to f16")
    print("-" * 40)
    baseline = results.get("f16", results["None"])
    for level, tps in results.items():
        rel = f"{(tps / baseline):.2f}x" if baseline else "-"
        print(f"{level:13} | {tps:5.1f} | {rel}")
