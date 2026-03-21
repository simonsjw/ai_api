
# ai_api

**Unified asynchronous Python library for local (Ollama) and remote (xAI Grok) LLMs**  
with batching, streaming, structured outputs, and optional PostgreSQL persistence.

## Features

- Single `create()` factory — identical interface for Grok and Ollama
- Async batch processing with semaphore-limited concurrency and exponential-backoff retries
- Full token-by-token streaming support
- Persistence modes: `"none"` (default), `"json_files"`, or `"postgres"` (daily-partitioned tables via `infopypg`)
- Structured JSON output (Pydantic models or raw JSON schemas)
- Grok conversation caching (`x-grok-conv-id`)
- Full type safety with immutable dataclasses (`LLMRequest`, `LLMResponse`, `StreamingChunk`)
- Provider-specific options and tool calling support

## Installation

```bash
git clone https://github.com/simonsjw/ai_api.git
cd ai_api
pip install -e ".[dev]"   # editable install (recommended for development)
```

**Requirements**  
- Python 3.12+  
- `infopypg` (custom library — install from your private repo or local path)  
- `python-dotenv` for `.env` loading

## Quick Start

```python
import asyncio
from pathlib import Path
from ai_api import create, LLMRequest
from ai_api.data_structures.LLM_types_grok import GrokInput

async def main():
    client = await create(
        provider="grok",                    # or "ollama"
        model="grok-2",                     # any supported model
        api_key="xai-...",                  # required for Grok
        # org="localhost",                  # for Ollama custom host
        save_mode="postgres",               # "none" | "json_files" | "postgres"
        # output_dir=Path("outputs"),
        concurrency=30,
        max_retries=3,
        timeout=60.0,
        set_conv_id=True,                   # enables Grok conversation caching
    )

    # Prepare request
    messages = [{"role": "user", "content": "Explain quantum computing in simple terms."}]
    grok_input = GrokInput.from_list(messages)

    req = LLMRequest(
        input=grok_input,
        model=client.model,
        temperature=0.7,
        # structured_schema=YourPydanticModel,   # enable JSON mode
    )

    # Batch mode
    responses = await client.submit_batch([req])
    print(responses[0].content if responses else "No response")

    # Or streaming
    async for chunk in client.stream(req):
        print(chunk.delta_text, end="", flush=True)

asyncio.run(main())
```

## Persistence Options

Set via `save_mode` in `create()`:

| Mode          | Description                              | Configuration Needed                  |
|---------------|------------------------------------------|---------------------------------------|
| `"none"`      | No persistence (fastest)                 | —                                     |
| `"json_files"`| Save raw JSON responses to disk          | `output_dir=Path(...)`                |
| `"postgres"`  | Daily-partitioned tables via `infopypg`  | Postgres credentials in `.env`        |

See `src/ai_api/data_structures/db_responses_schema.py` for the `DatabaseBuilder` example if you need to initialise tables manually.

## Providers

### Grok (xAI)
- Requires valid `api_key`
- Supports conversation caching and advanced Grok-specific options

### Ollama
- Works with any locally running model (`ollama serve`)
- Use `org` or `base_url` for custom endpoints


## Development

- Linting: `ruff`
- Type checking: `basedpyright` / `pyrefly`
- Tests: `pytest` (dev dependency — tests directory can be added later)

## License

MIT — see the `LICENSE` file (create it if missing using the standard MIT template).

