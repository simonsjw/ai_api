# `ai_api.core` — Unified LLM Client Framework

**The `core` package is the heart of `ai_api`.** It provides a clean, consistent, and extensible interface for interacting with multiple LLM providers while hiding all provider-specific complexity behind a unified API.

**New in 2026:** `ChatSession` is now the **recommended high-level API** for any conversation that needs branching, forking, or history editing. It wraps `PersistenceManager` + provider clients and maintains session state automatically.

## Philosophy

- **One API, many backends** — Swap `ollama` ↔ `xai` (or add new providers) with minimal code changes.
- **Git-style branching by default** — Every conversation has `tree_id` / `branch_id` / `parent_response_id` with zero history duplication.
- **History editing via rebase** — `edit_history()` creates a new branch with edits applied; the original branch stays immutable.
- **Protocol-driven design** — Structural subtyping (`LLMRequestProtocol`, `LLMResponseProtocol`) so new providers require almost no boilerplate.
- **Symmetrical persistence** — Every turn saved via the same `persist_chat_turn` path (Postgres or JSON).
- **Built-in structured output** — Pass any Pydantic model via `response_model=`.

## Directory Structure

```
core/
├── __init__.py
├── base_provider.py          # LLMProviderAdapter Protocol
├── client_factory.py         # get_llm_client() + registry
├── ollama_client.py          # Public Ollama facade
├── xai_client.py             # Public xAI facade
├── common/
│   ├── chat_session.py       # ★ ChatSession + edit_history (recommended)
│   ├── persistence.py        # PersistenceManager (create_edited_branch, reconstruct_neutral_branch)
│   ├── errors.py
│   └── response_struct.py
├── ollama/                   # Ollama-specific implementations
│   ├── chat_turn_ollama.py
│   ├── chat_stream_ollama.py
│   └── ...
└── xai/                      # xAI-specific
    ├── chat_turn_xai.py
    ├── chat_batch_xai.py
    └── ...
```

## Core Concepts

### 1. `ChatSession` — Recommended for Branched Conversations

```python
from ai_api.core.common.chat_session import ChatSession
from ai_api.core.client_factory import get_llm_client
from ai_api.core.common.persistence import PersistenceManager

pm = PersistenceManager(logger=logger, db_url="postgresql://...")
client = get_llm_client("ollama", persistence_manager=pm)
session = ChatSession(client, pm)

resp, meta = await session.create_or_continue("Hello world", model="llama3.2")
print(meta["tree_id"], meta["branch_id"])

# Continue
resp2, meta2 = await session.create_or_continue("How are you?")

# Edit history (creates new branch, session switches automatically)
result = await session.edit_history(
    edit_ops=[{"op": "remove_turns", "indices": [0]}],
    new_branch_name="without-greeting"
)
print("Now on branch:", session.current_branch_id)
```

**Key methods:**
- `create_or_continue(new_prompt, tree_id=..., branch_id=..., parent_response_id=..., **generation_kwargs)`
- `edit_history(edit_ops, new_branch_name=..., start_from_response_id=..., end_at_response_id=...)`

After `edit_history`, the session state (`current_branch_id`, `last_response_id`) is updated so subsequent calls continue from the edited history.

### 2. Clients (Lower-Level)

Use the raw clients only when you don't need branching state:

```python
client = get_llm_client("ollama", logger=logger, mode="stream", persistence_manager=pm)

async for chunk in client.create_chat(messages=[...], model="llama3.2", save_mode="postgres"):
    ...
```

### 3. Modes

| Mode     | Description                              | Notes |
|----------|------------------------------------------|-------|
| `turn`   | Single non-streaming completion          | Both providers |
| `stream` | Real-time token streaming                | Both |
| `batch`  | Process multiple conversations at once   | Native on xAI, simulated on Ollama |

### 4. Structured Output

Works in all modes, including per-request models in xAI batch:

```python
class Person(BaseModel):
    name: str
    age: int

response = await client.create_chat(..., response_model=Person)
print(response.parsed)          # validated instance
```

### 5. Persistence & Branching

`PersistenceManager` (in `common/persistence.py`) is the single source of truth:

- `persist_chat_turn(response, request, tree_id=..., branch_id=..., parent_response_id=...)`
- `reconstruct_neutral_branch(tree_id, branch_id, start_from_response_id=..., max_depth=...)` — fast recursive CTE
- `create_edited_branch(...)` — implements Git-style rebase (used by `ChatSession.edit_history`)
- `Conversations` table now tracks `active_branch_id` + `branch_metadata` (human-readable branch names)

**Zero duplication guarantee:** Only the delta (last prompt + generated response) is stored in the `response` JSONB column. Full history is reconstructed on demand.

### 6. Error Handling

All errors inherit from `AIAPIError`. Provider-specific errors are in `ollama/errors_ollama.py` and `xai/errors_xai.py`.

## Quick Start (Full Example with Editing)

```python
import logging
from ai_api.core.client_factory import get_llm_client
from ai_api.core.common.persistence import PersistenceManager
from ai_api.core.common.chat_session import ChatSession
from pydantic import BaseModel

logger = logging.getLogger(__name__)
pm = PersistenceManager(logger=logger, db_url="postgresql://...")

client = get_llm_client("ollama", logger=logger, persistence_manager=pm)
session = ChatSession(client, pm)

class Answer(BaseModel):
    summary: str
    confidence: float

# Turn 1
resp, meta = await session.create_or_continue(
    "Explain quantum entanglement",
    model="llama3.2",
    response_model=Answer
)

# Turn 2
await session.create_or_continue("What are practical applications?")

# Edit: remove first turn + add focus instruction
await session.edit_history(
    edit_ops=[
        {"op": "remove_turns", "indices": [0]},
        {"op": "insert_turn_after", "after_index": 0, "turn": {
            "role": "user", "content": "Focus only on the technical architecture."
        }},
    ],
    new_branch_name="technical-focus"
)

# Continue from cleaned history
await session.create_or_continue("What is the recommended tech stack?")
```

## Adding a New Provider

1. Create `core/newprovider/` with the chat modules.
2. Create `core/newprovider_client.py` with the three client classes.
3. Call `register_provider("newprovider", Turn..., Stream..., Batch...)` at the bottom.

That's it.

## Related Documentation

- `core/common/chat_session.py` — **Start here** for branching conversations
- `core/common/persistence.py` — `create_edited_branch`, `reconstruct_neutral_branch`, py_pgkit integration
- `core/client_factory.py` — Registry & unified factory
- `data_structures/` — Protocols and `Neutral*` models

All modules contain comprehensive NumPy-style docstrings with usage examples.

---

**`ai_api.core`** now makes branched, editable, production-grade LLM conversations simple and safe.