# `ai_api.data_structures` — Shared Data Models & Protocols

The `data_structures` package is the **foundational layer** of `ai_api`. It defines the contracts and concrete models that allow the entire library to remain provider-agnostic while still supporting rich, provider-specific functionality and powerful Git-style branching.

## Purpose

- Provide **minimal, stable protocols** (`LLMRequestProtocol`, `LLMResponseProtocol`, etc.)
- Supply **provider-specific concrete implementations** for Ollama and xAI
- Define the **Neutral* models** (`NeutralTurn`, `NeutralPrompt`, `NeutralResponseBlob`) used for zero-duplication persistence and history reconstruction
- Centralize `SaveMode` and structured-output specification classes
- Define the Postgres schema (`db_responses_schema.py`) including the `Conversations` table and branching columns

Everything in `core/` ultimately depends on the types defined here.

## Directory Structure

```
data_structures/
├── base_objects.py          # Core protocols + LLMEndpoint + SaveMode + Neutral* models
├── ollama_objects.py        # Ollama-specific request/response/streaming models
├── xai_objects.py           # xAI-specific request/response/streaming models
└── db_responses_schema.py   # SQLAlchemy declarative models for requests, responses, conversations, logs
```

## Core Concepts

### 1. Protocols (in `base_objects.py`)

| Protocol                    | Required Methods                          | Used By |
|-----------------------------|-------------------------------------------|---------|
| `LLMRequestProtocol`        | `meta()`, `payload()`, `endpoint()`       | `persist_chat_turn` |
| `LLMResponseProtocol`       | `meta()`, `payload()`, `endpoint()`, `to_neutral_format()` | `persist_chat_turn` + reconstruction |
| `LLMStreamingChunkProtocol` | `text`, `finish_reason`, `is_final`, ...  | Streaming clients |
| `LLMEndpoint`               | `provider`, `model`, `base_url`, ...      | Both request & response |

Any object satisfying these (via duck typing) can be persisted or passed through the system.

### 2. Neutral Models — The Heart of Branching & Zero Duplication

```python
from ai_api.data_structures.base_objects import (
    NeutralTurn, NeutralPrompt, NeutralResponseBlob
)
```

- `NeutralTurn` — role, content, structured, tools, finish_reason, usage, branch_meta, ...
- `NeutralPrompt` — system, user (last only), structured_spec, tools
- `NeutralResponseBlob` — prompt + response + branch_context

These are stored in the `response` JSONB column. Full history is **never duplicated** — only the delta for the current turn is saved. Reconstruction happens via `reconstruct_neutral_branch()` (recursive CTE on `parent_response_id`).

### 3. Provider-Specific Objects

- `ollama_objects.py` — `OllamaRequest`, `OllamaResponse`, `OllamaStreamingChunk`, embeddings, `OllamaJSONResponseSpec`
- `xai_objects.py` — `xAIRequest`, `xAIResponse`, `xAIStreamingChunk`, `xAIJSONResponseSpec`

They implement the protocols **and** add `from_neutral_history()`, `to_neutral_format()`, and provider-specific helpers.

### 4. `SaveMode`

```python
type SaveMode = Literal["none", "json_files", "postgres"]
```

Defined in `base_objects.py` and respected by `ChatSession` and `PersistenceManager`.

### 5. Database Schema (`db_responses_schema.py`)

Modern partitioned Postgres schema with Git-style branching:

- **`requests`** (legacy — now rarely written; request data folded into responses)
- **`responses`** (partitioned by `tstamp`)
  - `tree_id`, `branch_id`, `parent_response_id`, `sequence`
  - `response` JSONB (contains `NeutralResponseBlob`)
  - `meta` JSONB (merged request + response meta + kind, branching info)
- **`conversations`** — lightweight tree-level metadata
  - `tree_id`, `active_branch_id`, `title`, `meta`, `branch_metadata` (JSONB with human-readable branch names)
- **`logs`** (partitioned) — structured logging via py_pgkit
- **`providers`** — normalized lookup table

**Key design points:**
- Every row is **immutable** (edits = new branch with new `response_id`s)
- `reconstruct_neutral_branch(tree_id, branch_id, start_from=..., up_to=...)` walks the `parent_response_id` chain in a single fast query
- `create_edited_branch(...)` implements rebase semantics (used by `ChatSession.edit_history`)

## How It Fits Into the Bigger Picture

```
User Code
    │
    ▼
ChatSession (core/common/chat_session.py)
    │  create_or_continue / edit_history
    ▼
PersistenceManager (core/common/persistence.py)
    │  reconstruct_neutral_branch / create_edited_branch / persist_chat_turn
    ▼
data_structures/          ← You are here
    ├── Neutral* models + Protocols
    └── db_responses_schema (Conversations + branching columns)
    │
    ▼
Postgres (partitioned, immutable rows, recursive CTE reconstruction)
```

This separation is what makes branching safe and extensible: the persistence layer only cares about the protocol methods and the branching columns — it never needs to know Ollama vs xAI specifics.

## Key Files

| File                     | Role |
|--------------------------|------|
| `base_objects.py`        | Protocols + `NeutralTurn`/`NeutralResponseBlob` + `SaveMode` |
| `ollama_objects.py`      | All Ollama models + structured output spec + `from_neutral_history` |
| `xai_objects.py`         | All xAI models + structured output spec + `from_neutral_history` |
| `db_responses_schema.py` | `Requests`, `Responses`, `Conversations`, `Logs` — partitioned + branching |

## Usage Example (Neutral + Branching)

```python
from ai_api.data_structures.base_objects import LLMResponseProtocol
from ai_api.data_structures.ollama_objects import OllamaResponse

# After a turn, the response object implements to_neutral_format()
neutral = resp.to_neutral_format(branch_info={"tree_id": "...", "branch_id": "..."})

# Persisted as NeutralResponseBlob (only delta + branch pointers)
# Full history reconstructed on demand via reconstruct_neutral_branch()
```

## Migration Notes (from older versions)

- Table names changed from `llm_requests`/`llm_responses` → `requests`/`responses`
- `Requests` table is now largely unused (request data folded into `responses.meta` + `prompt`)
- New `Conversations` table + `active_branch_id` / `branch_metadata` columns
- All history editing now creates new immutable branches (no in-place mutation)

---

**`ai_api.data_structures`** provides the stable, minimal contracts that make provider-agnostic branching and history editing possible.