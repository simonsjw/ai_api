#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_reasoning.py
Demonstrates reasoning trace capture (include_reasoning) and effort levels
(low/medium/high) on both Grok and Ollama providers.

Updated 27 March 2026 to use current xAI models:
- grok-3-mini (supports effort levels)
- grok-4.20-0309-reasoning (capture-only reasoning variant)

Saves to PostgreSQL (automatic via LLMClient) and to JSON files (explicit).
"""

from __future__ import annotations

import asyncio
import json
import logging
from os import getenv
from pathlib import Path
from typing import Any

import debugpy

from ai_api.core.client import LLMClient
from ai_api.data_structures import (
    GrokInput,
    GrokRequest,
    OllamaInput,
    OllamaRequest,
)

# Create container for initial environmental variable container.
env_var: str | None

env_var = getenv("POSTGRES_DB_RESPONSES")
if env_var:
    db_settings: dict[str, str | list[str]] = json.loads(env_var)
else:
    raise PermissionError("Postgres data not provided.")

env_var = getenv("XAI_API_KEY")
if env_var:
    XAI_API_KEY: str = env_var
else:
    raise ("Grok API key not provided.")


async def run_demo(
    client: LLMClient,
    demo_logger: Logger,
    provider: str,
    model: str,
    question: str,
    include_reasoning: bool,
    reasoning_effort: str | None = None,
) -> None:
    """Run a single test case and persist results."""

    demo_logger.debug(
        "Demo settings",
        extra={
            "provider": provider,
            "model": model,
            "question": question,
            "include_reasoning": include_reasoning,
            "reasoning_effort": reasoning_effort,
        },
    )

    # Build input and request (payload creation point)
    if provider == "grok":
        input_obj: GrokInput = GrokInput.from_list(
            [{"role": "user", "content": question}]
        )
        request = GrokRequest(
            input=input_obj,
            model=model,
            include_reasoning=include_reasoning,
            reasoning_effort=reasoning_effort,
        )
    else:
        input_obj = OllamaInput.from_list([{"role": "user", "content": question}])
        request = OllamaRequest(
            input=input_obj,
            model=model,
            include_reasoning=include_reasoning,
            reasoning_effort=reasoning_effort,
        )

    print(
        f"\n=== {provider.upper()} | {model} | "
        f"include={include_reasoning} | effort={reasoning_effort} ==="
    )
    print(f"Question: {question}")

    # Generate (non-streaming)
    response = await client.generate(request, stream=False)

    # Extract results
    final_text = getattr(response, "text", "")
    reasoning_text = getattr(response, "reasoning_text", None)

    print(f"Final answer:\n{final_text}")
    if reasoning_text:
        print(
            f"Reasoning trace ({len(reasoning_text)} chars):\n{reasoning_text[:500]}..."
        )
    else:
        print("Reasoning trace: (none captured)")

        # === Save to PostgreSQL (automatic via client) ===
        # Already performed inside generate()

        # === Save to JSON file ===
    json_path = Path(
        f"reasoning_demo_{provider}_{model.replace(':', '_').replace('.', '_')}_"
        f"inc{int(include_reasoning)}_eff{reasoning_effort or 'None'}.json"
    )
    json_data = {
        "question": question,
        "provider": provider,
        "model": model,
        "include_reasoning": include_reasoning,
        "reasoning_effort": reasoning_effort,
        "final_text": final_text,
        "reasoning_text": reasoning_text,
        "full_raw_response": response.raw if hasattr(response, "raw") else None,
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))
    print(f"Saved to JSON: {json_path}")


async def main() -> None:
    # === Configuration (replace with your values) ===
    # db_settings is loaded at the start of this file along with the XAI API key.

    # logger.info("Info message.", extra={"obj": {"key": "value"}})
    from logger import Logger, setup_logger

    demo_logger: Logger = setup_logger(
        log_location=db_settings, log_level=logging.DEBUG
    )

    demo_logger.info("Logger initialised.")

    QUESTION = (
        "A bat and a ball cost $1.10 in total. "
        "The bat costs $1.00 more than the ball. "
        "How much does the ball cost? Show your reasoning step by step."
    )

    # Create ONE client per provider (reused for all tests)
    grok_client = LLMClient(
        provider="grok",
        model="grok-3-mini",                                                              # default; overridden per test
        settings=db_settings,
        api_key=XAI_API_KEY,
        logger=demo_logger,
    )

    ollama_client = LLMClient(
        provider="ollama",
        model="glm-4.7-flash:latest",                                                     # default; overridden per test
        settings=db_settings,
        logger=demo_logger,
    )

    # === Grok demonstrations (current models) ===
    await run_demo(
        grok_client, demo_logger, "grok", "grok-3-mini", QUESTION, True, "low"
    )
    await run_demo(
        grok_client, demo_logger, "grok", "grok-3-mini", QUESTION, True, "medium"
    )
    await run_demo(
        grok_client, demo_logger, "grok", "grok-3-mini", QUESTION, True, "high"
    )
    await run_demo(
        grok_client, demo_logger, "grok", "grok-3-mini", QUESTION, False, None
    )
    await run_demo(
        grok_client,
        demo_logger,
        "grok",
        "grok-4.20-0309-reasoning",
        QUESTION,
        True,
        None,
    )

    # === Ollama demonstrations (your local models) ===
    await run_demo(
        ollama_client,
        demo_logger,
        "ollama",
        "glm-4.7-flash:latest",
        QUESTION,
        True,
        "low",
    )
    await run_demo(
        ollama_client,
        demo_logger,
        "ollama",
        "glm-4.7-flash:latest",
        QUESTION,
        True,
        "medium",
    )
    await run_demo(
        ollama_client,
        demo_logger,
        "ollama",
        "glm-4.7-flash:latest",
        QUESTION,
        True,
        "high",
    )
    await run_demo(
        ollama_client,
        demo_logger,
        "ollama",
        "glm-4.7-flash:latest",
        QUESTION,
        False,
        None,
    )

    await run_demo(
        ollama_client,
        demo_logger,
        "ollama",
        "qwen3-coder-next:latest",
        QUESTION,
        True,
        None,
    )


if __name__ == "__main__":
    asyncio.run(main())
