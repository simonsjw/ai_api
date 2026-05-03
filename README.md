# ai_api

**A clean, unified, and extensible Python interface for interacting with local and remote Large Language Models — with first-class Git-style branching and history editing.**

`ai_api` gives you a consistent API across providers (Ollama, xAI, and easily extensible to others) while handling persistence, structured output, streaming, batching, error handling, **and powerful conversation branching** out of the box.

---

## ✨ Key Features

- **Unified API** — Same `create_chat()` (or `ChatSession`) interface for Ollama and xAI
- **Git-Style Branching** — Full `tree_id` / `branch_id` / `parent_response_id` model with zero history duplication
- **History Editing (Rebase)** — `edit_history()` lets you remove, insert, or replace turns and create a new branch (original stays immutable)
- **Multiple Modes** — `turn`, `stream`, and native `batch` (xAI)
- **Structured Output** — Pass any Pydantic model via `response_model=` 
- **Symmetrical Persistence** — Save to Postgres (recommended) or JSON files using `PersistenceManager`
- **Protocol-Driven Design** — Minimal contracts make adding new providers trivial
- **Rich Error Handling** — Consistent exception hierarchy
- **Model Management** — Ollama + xAI catalog support

---

## 🚀 Quick Start (Recommended: ChatSession)

```python
import logging
from ai_api.core.client_factory import get_llm_client
from ai_api.core.common.persistence import PersistenceManager
from ai_api.core.common.chat_session import ChatSession
from pydantic import BaseModel

logger = logging.getLogger(__name__)
pm = PersistenceManager(logger=logger, db_url="postgresql://user:pass@localhost/ai_logs")

client = get_llm_client("ollama", logger=logger, persistence_manager=pm)

class Answer(BaseModel):
    summary: str
    confidence: float

session = ChatSession(client, pm)

# Normal conversation (creates tree + branch automatically)
resp, meta = await session.create_or_continue(
    "Explain quantum entanglement briefly",
    model="llama3.2",
    response_model=Answer
)
print(resp.parsed.summary)

# Continue the conversation
resp2, meta2 = await session.create_or_continue("What are the practical applications?")

# Later: edit history (remove first turn, add clarifying instruction)
result = await session.edit_history(
    edit_ops=[
        {"op": "remove_turns", "indices": [0]},
        {"op": "insert_turn_after", "after_index": 0, "turn": {
            "role": "user",
            "content": "Focus only on the technical architecture."
        }},
    ],
    new_branch_name="technical-focus"
)

print("Now on new branch:", session.current_branch_id)

# Continue from the edited history
resp3, meta3 = await session.create_or_continue("What is the recommended tech stack?")
```

---

## 🏗️ Architecture Overview

```
    ┌─────────────────────────────────────────────────────────────┐
    │                      Your Application                       │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  core/client_factory │  ← get_llm_client()
                        └──────────┬───────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
             ollama_client    xai_client     (future)
                    │              │
                    ▼              ▼
             core/common/          ← ChatSession, PersistenceManager
             (chat_session.py, persistence.py)
                    │
                    ▼
             data_structures/      ← Protocols + Neutral* models
                    │
                    ▼
             Postgres (requests, responses, conversations tables)
             or JSON files
```

**Three clean layers:**

- **`core/`** — Public clients, factory, `ChatSession` (recommended for branching)
- **`data_structures/`** — Foundational protocols (`LLMRequestProtocol`, etc.) + `NeutralTurn` / `NeutralResponseBlob`
- **`common/`** — `PersistenceManager` (now with `create_edited_branch`), errors, response structuring

---

## 📚 Documentation

| Document                        | Purpose                                              | Location |
|--------------------------------|------------------------------------------------------|----------|
| **This README**                | High-level overview, branching & editing examples    | Root |
| **`core/core_README.md`**      | `ChatSession`, clients, factory, modes, persistence  | `src/ai_api/core/` |
| **`data_structures/data_structures_README.md`** | Protocols, Neutral models, DB schema, branching     | `src/ai_api/data_structures/` |

All Python modules contain rich NumPy-style docstrings.

---

## Supported Providers

| Provider | Status     | Strengths                              | Unique Features                     |
|----------|------------|----------------------------------------|-------------------------------------|
| **Ollama**   | ✅ Full    | Local, fast, private                   | Native embeddings, model management |
| **xAI**      | ✅ Full    | Powerful Grok models, native batch     | Per-request structured output, rich errors |

Adding a new provider takes only a few files + one `register_provider()` call.

---

## 📦 Installation

```bash
pip install ai-api
# or from source
pip install -e .
```

**Requirements:** Python 3.10+, `httpx`, `pydantic`, `asyncpg` (optional but recommended for Postgres).

---

## Branching & History Editing (The Killer Feature)

`ai_api` uses a **Git-style model**:

- Every conversation lives under a `tree_id`
- Each linear path is a `branch_id`
- Turns are linked via `parent_response_id` + `sequence`
- Full history is **never duplicated** — only deltas are stored; reconstruction uses a fast recursive CTE

### Example 1: Normal Conversation + Fork

```python
session = ChatSession(client, pm)

await session.create_or_continue("Tell me about the project")
await session.create_or_continue("What are the main challenges?")

# Fork from the second turn
resp, meta = await session.create_or_continue(
    "How do we solve the scaling issue?",
    parent_response_id=meta["last_response_id"]   # or just omit to continue
)
```

### Example 2: Edit History (Rebase) — Recommended Pattern

```python
# After a long conversation, clean it up
result = await session.edit_history(
    edit_ops=[
        {"op": "remove_turns", "indices": [0, 1]},           # drop irrelevant intro
        {"op": "insert_turn_after", "after_index": 2, "turn": {
            "role": "user",
            "content": "Please focus only on the technical architecture."
        }},
        {"op": "replace_turn", "index": 5, "turn": {
            "role": "assistant",
            "content": "Corrected response..."
        }},
    ],
    new_branch_name="clean-technical-v2"
)

# Session automatically switches to the new branch
print("Active branch:", session.current_branch_id)

# Continue from the cleaned history
await session.create_or_continue("What is the final recommended stack?")
```

The original branch remains **completely immutable** — you can always go back to it.

---

## Why ai_api?

- **Consistency** — Same code for local and remote models
- **Production Ready** — Built-in persistence, logging, branching, and error handling
- **Extensible** — Protocol-based design
- **Developer Friendly** — Excellent docs + `ChatSession` hides all complexity

---

## Contributing

Contributions welcome! Focus areas: new providers, improved branching UX, caching for very long conversations, additional persistence backends.

---

**ai_api** — One clean interface. Any LLM. Built for real applications with real conversation history.