# ai_api: Asynchronous Client for xAI Grok API with Persistence

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

ai_api is a modular Python library for interacting with the xAI Grok API, focusing on asynchronous batched requests, retry logic, and optional persistence of requests and responses. It supports saving to local JSON files or a PostgreSQL database, leveraging the custom `infopypg` library for efficient database operations (e.g., connection pooling and incremental setup). The module is structured into two subpackages: `data_structures` for type definitions and database schemas, and `grok_client` for the asynchronous API client.

Key features:
- **Asynchronous Batched Requests**: Uses `aiohttp` for concurrent API calls to the Grok chat completions endpoint, with configurable concurrency, timeouts, and retries with exponential backoff.
- **Persistence Options**: No saving ("none"), JSON files ("json_files"), or PostgreSQL ("postgres") via `infopypg`. Database mode uses partitioned tables for scalability (e.g., daily ranges on timestamps).
- **Type-Safe Models**: TypedDicts and dataclasses for OpenAI-compatible and Grok-specific request/response structures, ensuring structural consistency.
- **Logging Integration**: Utilises a custom `logger` module for structured logging to files or PostgreSQL, with automatic setup and lazy initialisation.
- **Database Schema**: Normalised tables for providers, requests, responses, and logs, defined via SQLAlchemy models in `data_structures`.
- Designed for the `grok` conda environment; assumes dependencies like `aiohttp`, `asyncpg`, `infopypg`, and `logger` are available.

This library prioritises efficiency (e.g., shared connection pools from `infopypg`) and modularity, avoiding heavy ORM overhead while supporting scalable data capture for LLM interactions. It assumes the xAI API follows OpenAI-compatible schemas and integrates with `infopypg` for PostgreSQL setup (e.g., incremental creation of tablespaces, databases, extensions, and tables).

## Installation

This package is intended for use within the `grok` conda environment (as defined in `environment_grok.yml`). To install as an editable package:

1. Navigate to the project root (where `pyproject.toml` resides).
2. Activate the `grok` environment:
   ```
   conda activate grok
   ```
3. Install via pip:
   ```
   pip install -e .
   ```

This makes the `ai_api` module importable in your scripts.

### Dependencies

- Python 3.12+
- `aiohttp` (for asynchronous HTTP requests)
- `asyncpg` (for PostgreSQL interactions, via `infopypg`)
- `SQLAlchemy>=2.0` (for declarative database models)
- Custom `infopypg` library (for PostgreSQL pooling, setup, and queries; see [README_infopypg.md](https://github.com/simonsjw/infopypg/blob/master/README_infopypg.md))
- Custom `logger` module (for structured logging to files or PostgreSQL; see [README_logger.md](https://github.com/simonsjw/logger/blob/master/README_logger.md))
Full environment details are in `environment_grok.yml`. Linting and type-checking are configured via `pyproject.toml` (using `ruff` and `basedpyright`).

## Usage

### Data Structures

The `data_structures` subpackage provides types and schemas:

```python
from ai_api.data_structures import GrokRequest, GrokMessage, GrokResponse, responses_default_settings, Providers, Responses

# Example request
messages = [GrokMessage(role="user", content="Hello, Grok!")]
req = GrokRequest(messages=messages, model="grok-beta")
```

For PostgreSQL schemas, use SQLAlchemy models like `Providers` and `Responses` with `infopypg`'s `DatabaseBuilder` for setup.

### Grok Client

The `grok_client` subpackage provides the asynchronous client:

```python
import asyncio
from pathlib import Path
from ai_api.grok_client import XAIAsyncClient
from ai_api.data_structures import GrokRequest, GrokMessage
from infopypg.pgtypes import ResolvedSettingsDict

# Example with JSON saving
client = XAIAsyncClient(
    api_key="xai_...",
    save_mode="json_files",
    output_dir=Path("outputs"),
    concurrency=50,
    timeout=60.0,
    max_retries=3,
    set_conv_id=True,  # Optional for caching
)

messages = [GrokMessage(role="user", content="Hello, Grok!")]
req = GrokRequest(messages=messages)
results = await client.submit_batch([req])
if results:
    print(results[0].content)  # Prints the response content

# Example with PostgreSQL saving (using infopypg ResolvedSettingsDict)
pg_settings: ResolvedSettingsDict = {
    "DB_USER": "postgres",
    "DB_HOST": "127.0.0.1",
    "DB_PORT": "5432",
    "DB_NAME": "responses_db",
    "PASSWORD": "your_password",
    "TABLESPACE_NAME": "responses_db",
    "TABLESPACE_PATH": "/mnt/HDD03_HIT_03TB/no_backup/pg03/responses_db",
    "EXTENSIONS": ["uuid-ossp", "pg_trgm"],
}

client_pg = XAIAsyncClient(
    api_key="xai_...",
    save_mode="postgres",
    resolved_pg_settings=pg_settings,
)

await client_pg.submit_batch([req], return_responses=False)  # Saves to DB, returns None
```

For PostgreSQL mode, ensure the database is set up using `infopypg`'s `DatabaseBuilder` with the schema from `db_responses_schema.py` (e.g., partitioned tables for requests/responses/logs).

### Logging

Logging is handled via the `logger` module, initialised in `grok_client` with `setup_logger`. For PostgreSQL logging, pass a `ResolvedSettingsDict` to `setup_logger` as described in [README_logger.md](README_logger.md). Queries can be executed using `infopypg`'s `execute_query` or the logger's `query_logs`.

## Configuration Options

- **XAIAsyncClient Parameters**:
  - `api_key`: Required xAI API key.
  - `save_mode`: "none" (default), "json_files", or "postgres".
  - `resolved_pg_settings`: Required for "postgres" (from `infopypg`).
  - `output_dir`: Required for "json_files".
  - `concurrency`: Max concurrent requests (default: 50).
  - `timeout`: Request timeout in seconds (default: 60.0).
  - `max_retries`: Retry attempts (default: 3).
  - `set_conv_id`: Enable conversation ID for caching (default: False).

For database setup, use `infopypg`'s tools to validate and resolve settings (e.g., `validate_dict_to_settings`, `resolve_postgres_connection_settings`).

## Examples

### Batched Requests with Persistence

```python
requests = [GrokRequest(messages=[GrokMessage(role="user", content=f"Query {i}")]) for i in range(10)]
await client.submit_batch(requests)  # Processes batch asynchronously
```

### Database Setup

```python
import asyncio
from infopypg import DatabaseBuilder
from ai_api.data_structures import responses_default_settings  # Default ResolvedSettingsDict

async def setup_db():
    builder = DatabaseBuilder(
        spec_path="src/ai_api/data_structures/db_responses_schema.py",
        resolved_settings=responses_default_settings,
    )
    await builder.build()  # Creates tablespace, DB, extensions, and tables if missing

asyncio.run(setup_db())
```

### Sequential Chaining for Caching

To enable caching with `set_conv_id=True`, process similar prompts sequentially:

```python
import asyncio
from ai_api.grok_client import XAIAsyncClient
from ai_api.data_structures import GrokRequest, GrokMessage

async def main():
    client = XAIAsyncClient(
        api_key="xai_...",
        save_mode="none",  # Or your preferred mode
        set_conv_id=True,  # Enables shared conv_id for caching
    )

    # Similar prompts (shared prefix for cache hit)
    base_msg = "Analyse the following text: "
    prompts = [
        base_msg + "The quick brown fox jumps over the lazy dog.",
        base_msg + "The quick brown fox jumps over the lazy cat.",
        base_msg + "The quick brown fox jumps over the lazy rabbit.",
    ]

    responses = []
    for prompt_text in prompts:
        messages = [GrokMessage(role="user", content=prompt_text)]
        req = GrokRequest(messages=messages, model="grok-beta")
        result = await client.submit_batch([req])  # Sequential await
        if result:
            responses.append(result[0])
            print(f"Response: {result[0].content}")

asyncio.run(main())

```

For multiple similar prompts (e.g., sharing a prefix like "Analyse the following text: "), sequential chaining allows the first response to populate Grok's cache. Subsequent prompts hit this cache, reducing token usage and costs (potentially by 50-90% for overlapping text, per xAI docs). This is efficient for workflows like batch analysis of variants, avoiding parallel race conditions where no cache exists yet.


## Development and Structure

- **Source Layout**:
  ```
  ai_api/
  ├── src/
  │   └── ai_api/
  │       ├── data_structures/
  │       │   ├── __init__.py      # Exports types and schemas
  │       │   ├── LLM_types.py     # TypedDicts and dataclasses for API structures
  │       │   └── db_responses_schema.py  # SQLAlchemy models for PostgreSQL
  │       └── grok_client/
  │           └── grok_client.py   # Asynchronous client implementation
  ├── pyproject.toml               # Config for ruff, basedpyright, etc.
  └── README.md                    # This file
  ```

- **Linting/Type-Checking**: Run `ruff check` and `basedpyright` from the root.
- **Testing**: Add tests in a future `tests/` directory; currently, rely on examples in docstrings.

## Full List of Exposed Functionality

All elements can be imported from `ai_api.data_structures` or `ai_api.grok_client`.

From `data_structures`:
- Types: `OPEN_AI_*`, `SaveMode`, `GrokMessage`, `GrokRequest`, `GrokResponse`
- Schema: `responses_default_settings`, `Providers`, `Requests`, `Responses`, `Logs`

From `grok_client`:
- `XAIAsyncClient`: Main client class with `submit_batch` method.

## Contributing

Contributions are welcome! Follow the code style: Australian English, 88-column lines, full type hints (e.g., using `|` for unions), NumPy-style docstrings. Ensure compatibility with Python 3.12 and the `grok` environment. Use `infopypg` and `logger` for database and logging to avoid duplication.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

For questions or issues, contact the maintainer. Last updated: January 06, 2026.
