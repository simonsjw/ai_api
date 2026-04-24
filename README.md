# ai_api

**A clean, unified, and extensible Python interface for interacting with local and remote Large Language Models.**

`ai_api` gives you a consistent API across providers (Ollama, xAI, and easily extensible to others) while handling persistence, structured output, streaming, batching, and error handling out of the box.

---

## ✨ Key Features

- **Unified API** — Same `create_chat()` interface for Ollama and xAI (swap providers with one line)
- **Multiple Modes** — `turn` (non-streaming), `stream` (real-time tokens), and `batch` (native on xAI)
- **Structured Output** — Pass any Pydantic model via `response_model=` and get validated objects back
- **Symmetrical Persistence** — Automatically save requests and responses to Postgres or JSON files
- **Protocol-Driven Design** — Minimal contracts (`LLMRequestProtocol`, `LLMResponseProtocol`) make adding new providers trivial
- **Rich Error Handling** — Consistent exception hierarchy with full backward compatibility
- **Model Management** — List, pull, and inspect models (Ollama) or query catalog (xAI)

---

## 🚀 Quick Start

```python
from ai_api.core.client_factory import get_llm_client
from ai_api.core.common.persistence import PersistenceManager
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
pm = PersistenceManager(logger=logger, db_url="postgresql://user:pass@localhost/ai_logs")

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
    messages=[{"role": "user", "content": "Explain quantum entanglement briefly"}],
    model="llama3.2",            # or "grok-4"
    response_model=Answer,
    save_mode="postgres"
):
    print(chunk.text, end="")
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Application                        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   core/client_factory │  ← Unified entry point
                    │   get_llm_client()    │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
         ollama_client    xai_client     (future providers)
                │              │
                ▼              ▼
         core/ollama/    core/xai/     ← Provider-specific logic
                │              │
                └──────┬───────┘
                       ▼
              ai_api.data_structures     ← Protocols & models
              (LLMRequestProtocol, LLMResponseProtocol, etc.)
                       │
                       ▼
              core/common/               ← Shared utilities
              (PersistenceManager, errors, response_struct)
```

**Three clean layers:**

- **`core/`** — Public clients, factory, and orchestration
- **`data_structures/`** — Foundational protocols and provider-specific models
- **`common/`** — Persistence, error handling, and structured output helpers

---

## 📚 Documentation

| Document                        | Purpose                                      | Location |
|--------------------------------|----------------------------------------------|----------|
| **This README**                | High-level overview & quick start            | Root |
| **`core/README.md`**           | Detailed guide to clients, factory & modes   | `src/ai_api/core/` |
| **`data_structures/README.md`** | Protocols, models, and extensibility         | `src/ai_api/data_structures/` |

All Python modules also contain rich NumPy-style docstrings with usage examples.

---

## Supported Providers

| Provider | Status     | Strengths                              | Unique Features                     |
|----------|------------|----------------------------------------|-------------------------------------|
| **Ollama**   | ✅ Full    | Local, fast, private                   | Native embeddings, model management, many generation params |
| **xAI**      | ✅ Full    | Powerful Grok models, native batch     | Per-request structured output in batch, rich remote errors |

Adding a new provider (Groq, Anthropic, OpenAI, vLLM, etc.) takes only a few files and one registration call.

---

## 📦 Installation

```bash
pip install ai-api
# or from source
pip install -e .
```

**Requirements:** Python 3.10+, `httpx`, `pydantic`, `asyncpg` (optional, for Postgres persistence).

---

## Example: Batch + Structured Output (xAI)

```python
results = await client.create_chat(
    messages_list=[conv1, conv2, conv3],
    model="grok-4",
    response_model=[Person, Summary, None],   # different model per item
    save_mode="json_files"
)
```

---

## Why ai_api?

- **Consistency** — Same code works for local and remote models
- **Production Ready** — Built-in persistence, logging, and error handling
- **Extensible** — Protocol-based design makes adding providers painless
- **Developer Friendly** — Excellent documentation and clear separation of concerns

---

## Contributing

Contributions are welcome! Please open an issue or pull request. Focus areas:

- New provider integrations
- Additional persistence backends
- Improved streaming & batch performance
- Documentation improvements

---

**ai_api** — One clean interface. Any LLM. Built for real applications.

*Made with care for developers who want power without the boilerplate.*