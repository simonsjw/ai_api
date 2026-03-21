# ai_api: Asynchronous Client for xAI Grok API with Persistence

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

`ai_api` is a modular Python library for interacting with the xAI Grok API (`/v1/responses`), with a focus on:

- asynchronous batched requests
- configurable concurrency and automatic retries
- optional persistence of requests and responses (JSON files or PostgreSQL via `infopypg`)
- type-safe, immutable dataclasses for requests and responses

It is designed to work efficiently in the `grok` conda environment and integrates tightly with the custom `infopypg` and `logger` modules.

Key features:

- Concurrent API calls using `aiohttp` + semaphore-limited parallelism
- Exponential backoff retries for transient failures
- Persistence modes: `"none"`, `"json_files"`, `"postgres"` (partitioned tables)
- Immutable dataclasses: `GrokMessage`, `GrokInput`, `GrokRequest`, `GrokResponse`
- Structured JSON output support via Pydantic models or dict schemas
- Conversation-level caching support via `x-grok-conv-id` header
- Structured logging to file or PostgreSQL

## Installation

Activate your `grok` conda environment and install the package in editable mode:

```bash
conda activate grok
cd /path/to/ai_api
pip install -e .
```

Dependencies are listed in `environment_grok.yml`.

## Usage

### Creating Requests

Requests are now built using `GrokInput` (which wraps an immutable tuple of `GrokMessage` instances).

```python
from ai_api.data_structures import GrokMessage, GrokInput, GrokRequest

# Basic text request
messages = [{"role": "user", "content": "Hello, Grok!"}]
grok_input = GrokInput.from_list(messages)
req = GrokRequest(input=grok_input, model="grok-3")

# Multimodal example
messages = [
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Describe this image."},
            {"type": "input_image", "image_url": "https://example.com/img.jpg", "detail": "high"},
        ],
    }
]
grok_input = GrokInput.from_list(messages)
req = GrokRequest(input=grok_input)
```

### Basic Client Usage

```python
import asyncio
from pathlib import Path
from ai_api.grok_client import XAIAsyncClient
from ai_api.data_structures import GrokInput, GrokMessage, GrokRequest
from infopypg.pgtypes import ResolvedSettingsDict

async def main():
    client = await XAIAsyncClient(
        api_key="xai_...",
        save_mode="json_files",
        output_dir=Path("outputs"),
        concurrency=50,
        timeout=60.0,
        max_retries=3,
        set_conv_id=True,           # recommended for cost reduction on similar prompts
    )

    # Single request
    messages = [{"role": "user", "content": "What is the capital of Australia?"}]
    grok_input = GrokInput.from_list(messages)
    req = GrokRequest(input=grok_input)
    results = await client.submit_batch([req])
    
    return results

results = asyncio.run(main())

if results:
    print(results[0].text)                  # → "The capital of Australia is Canberra."
    print(results[0].usage)                 # token counts
```

### Structured Output (JSON mode)

```python
from pydantic import BaseModel, Field

class MathAnswer(BaseModel):
    value: int = Field(description="The numerical result")
    reasoning: str = Field(description="Step-by-step explanation")

messages = [{"role": "user", "content": "What is 17 × 42? Return JSON."}]
grok_input = GrokInput.from_list(messages)

req = GrokRequest(
    input=grok_input,
    structured_schema=MathAnswer,    # this instructs the model to return the content in JSON. 
    temperature=0.1,
    max_output_tokens=300,
)

async def main():
    client = await XAIAsyncClient(
        api_key="xai_...",
        save_mode="json_files", # save response to JSON file, not return the content in JSON. 
        output_dir=Path("outputs"),
        concurrency=50,
        timeout=60.0,
        max_retries=3,
        set_conv_id=True,           # recommended for cost reduction on similar prompts
    )
    results = await client.submit_batch([req])
    
    return results
    
results = asyncio.run(main())

if results:
    print(results[0].text)                  # JSON string
    # parsed = MathAnswer.model_validate_json(results[0].text)
```

### PostgreSQL Persistence

```python
from infopypg.pgtypes import ResolvedSettingsDict

pg_settings: ResolvedSettingsDict = {
    "DB_USER": "simon",
    "DB_HOST": "localhost",
    "DB_PORT": "5432",
    "DB_NAME": "grok_responses",
    "PASSWORD": "...",
    # ... other infopypg settings
}

async def main():
    client_pg = await XAIAsyncClient(
    api_key="xai_...",
    save_mode="postgres",
    resolved_pg_settings=pg_settings,
    set_conv_id=True,
    )
    await client_pg.submit_batch([req], return_responses=False)

# Save only — no results returned
asyncio.run(main())
```

### Sequential Chaining for Cache Reuse

To get the maximum benefit from Grok's conversation caching use the below approach. 
The first prompt populates the cache; later similar prompts should reuse tokens for the shared prefix.

```python
async def main():
    client = await XAIAsyncClient(
        api_key="xai_...",
        save_mode="json_files",
        output_dir=Path("cache_test"),
        set_conv_id=True,                    # same conv_id across calls
    )

    base = "Carefully analyse this sentence: "
    variants = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox leaps over the lazy cat.",
        "The quick brown fox bounds over the sleepy rabbit.",
    ]

    for text in variants:
        messages = [{"role": "user", "content": base + text}]
        inp = GrokInput.from_list(messages)
        req = GrokRequest(input=inp, temperature=1.1, max_output_tokens=400)

        results = await client.submit_batch([req])
        if results:
            print(f"→ {results[0].text[:80]}…")

asyncio.run(main())
```


## Database Setup

Use `infopypg.DatabaseBuilder` to create the schema:

```python
from infopypg import DatabaseBuilder
from ai_api.data_structures import responses_default_settings

async def setup():
    builder = DatabaseBuilder(
        spec_path="src/ai_api/data_structures/db_responses_schema.py",
        resolved_settings=responses_default_settings,
    )
    await builder.build()

asyncio.run(setup())
```

This creates partitioned tables (`requests`, `responses`, etc.) with daily ranges.

## Project Structure

```
ai_api/
├── src/
│   └── ai_api/
│       ├── data_structures/
│       │   ├── __init__.py
│       │   ├── LLM_types_grok.py     # GrokMessage, GrokInput, GrokRequest, GrokResponse
│       │   └── db_responses_schema.py
│       └── grok_client/
│           └── grok_client.py
├── pyproject.toml
└── README.md
```

## Contributing

- Use Australian English
- Line width ≤ 88 columns
- Full type hints (`|` unions preferred)
- NumPy-style docstrings
- Functions ≤ 40 lines where practical
- Prioritise: efficiency → clarity of ideas → readability

Run `ruff check` and type checking before committing.

## License

MIT. See [LICENSE](LICENSE).

Last updated: January 2026
