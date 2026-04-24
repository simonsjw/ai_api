# `ai_api.core` — Unified LLM Client Framework

**The `core` package is the heart of `ai_api`.** It provides a clean, consistent, and extensible interface for interacting with multiple LLM providers (currently Ollama and xAI) while hiding all provider-specific complexity behind a unified API.

## Philosophy

- **One API, many backends** — Swap `ollama` ↔ `xai` (or add new providers) with minimal code changes.
- **Protocol-driven design** — Uses structural subtyping (`LLMProviderAdapter`, `LLMRequestProtocol`, `LLMResponseProtocol`) so new providers require almost no boilerplate.
- **Symmetrical persistence** — Every request and response can be saved to Postgres or JSON files using the same code path.
- **Built-in structured output** — Pass any Pydantic model via `response_model=` and get validated objects back.
- **Factory + self-registration** — Adding a new provider is literally three lines of code.

## Directory Structure

```
core/
├── __init__.py
├── base_provider.py          # LLMProviderAdapter Protocol (structural typing)
├── client_factory.py         # Unified get_llm_client() + registry
├── ollama_client.py          # Public Ollama facade + factory function
├── xai_client.py             # Public xAI facade + factory function
├── common/                   # Shared utilities used by ALL providers
│   ├── errors.py
│   ├── persistence.py
│   └── response_struct.py
├── ollama/                   # Ollama-specific implementations
│   ├── chat_turn_ollama.py
│   ├── chat_stream_ollama.py
│   ├── embeddings_ollama.py
│   └── errors_ollama.py
└── xai/                      # xAI-specific implementations
    ├── chat_turn_xai.py
    ├── chat_stream_xai.py
    ├── chat_batch_xai.py
    └── errors_xai.py
```

## Core Concepts

### 1. Clients (Public Entry Points)

Use the high-level clients exactly the same way regardless of provider:

```python
from ai_api.core.ollama_client import OllamaClient
from ai_api.core.xai_client import XAIClient
# or the unified factory
from ai_api.core.client_factory import get_llm_client

client = get_llm_client("ollama", logger=logger, mode="stream", host=...)
# or
client = OllamaClient(logger=logger, mode="turn", persistence_manager=pm)
```

Every client exposes:

- `create_chat(messages, model, mode=..., response_model=..., save_mode=...)`
- `aclose()` for graceful shutdown
- Provider-specific extras (Ollama: `pull_model()`, `get_model_options()`; xAI: `list_models()`, `get_model_info()`)

### 2. Modes

All providers support three interaction modes via the `mode` parameter or dedicated client classes:

| Mode     | Description                              | Unique to          |
|----------|------------------------------------------|--------------------|
| `turn`   | Single non-streaming completion          | Both               |
| `stream` | Real-time token streaming                | Both               |
| `batch`  | Process multiple conversations at once   | **xAI native**<br>Ollama simulated |

### 3. Structured Output (`response_model`)

Pass any Pydantic `BaseModel`:

```python
class Person(BaseModel):
    name: str
    age: int

response = await client.create_chat(
    messages=[...],
    model="llama3.2",           # or "grok-4"
    response_model=Person
)
print(response.parsed)   # → validated Person instance
```

Works in **all modes** (including per-request models in xAI batch).

### 4. Persistence (`save_mode`)

```python
response = await client.create_chat(
    ...,
    save_mode="postgres"        # or "json_files" or "none"
)
```

Uses the symmetrical `PersistenceManager` from `common/persistence.py`. Both requests and responses are stored using the same protocol interface.

### 5. Error Handling

All errors inherit from `AIAPIError` (in `common/errors.py`). Provider-specific errors live in `ollama/errors_ollama.py` and `xai/errors_xai.py` and are fully backward-compatible.

## Quick Start

```python
import logging
from ai_api.core.client_factory import get_llm_client
from ai_api.core.common.persistence import PersistenceManager
from pydantic import BaseModel

logger = logging.getLogger(__name__)
pm = PersistenceManager(logger=logger, db_url="postgresql://...")

client = get_llm_client(
    "ollama",                    # or "xai"
    logger=logger,
    mode="stream",
    persistence_manager=pm
)

class Answer(BaseModel):
    summary: str
    confidence: float

async for chunk in client.create_chat(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    model="llama3.2",
    response_model=Answer,
    save_mode="postgres"
):
    print(chunk.text, end="")
```

## Provider Comparison

| Feature                    | Ollama                              | xAI                                      |
|---------------------------|-------------------------------------|------------------------------------------|
| **Transport**             | Native HTTP (`httpx`)               | Official SDK (gRPC + HTTP)               |
| **Batch**                 | Simulated (sequential/concurrent)   | **Native** + per-request `response_model` lists |
| **Embeddings**            | First-class (`/api/embed`)          | Not yet implemented                      |
| **Model Management**      | `pull_model()`, `show_model()`, `get_model_options()` | `list_models()`, `get_model_info()`     |
| **Generation Params**     | Very rich (`num_ctx`, `repeat_penalty`, `think`, `mirostat`...) | Standard + thinking mode                 |
| **Error Types**           | HTTP + GPU memory warning           | Rich (rate limit, auth, multimodal, cache, batch) |
| **Local / Remote**        | Local (localhost:11434)             | Remote (api.x.ai)                        |

## Adding a New Provider

1. Create `core/newprovider/` with `chat_turn_*.py`, `chat_stream_*.py`, `chat_batch_*.py` (and `errors_*.py` if needed).
2. Create `core/newprovider_client.py` with `Turn*Client`, `Stream*Client`, `Batch*Client` classes + factory function.
3. At the bottom of `newprovider_client.py` call:

```python
from .client_factory import register_provider
register_provider("newprovider", TurnNewProviderClient, StreamNewProviderClient, BatchNewProviderClient)
```

That's it — `get_llm_client("newprovider", ...)` now works everywhere.

## Related Documentation

- `core/base_provider.py` — The structural protocol
- `core/client_factory.py` — Registry & unified factory
- `core/ollama_client.py` & `core/xai_client.py` — Public client facades
- `core/common/` — Persistence, errors, structured output (used by everyone)
- `core/ollama/` & `core/xai/` — Provider-specific implementations

All modules contain comprehensive NumPy-style docstrings with usage examples.

---

**`ai_api.core`** gives you a production-ready, extensible, and beautifully consistent interface to any LLM — today and tomorrow.