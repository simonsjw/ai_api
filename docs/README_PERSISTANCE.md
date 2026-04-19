# Conversation Branching & History Persistence – xAIClient

## Overview

The `xAIPersistenceManager` provides full **Git-style conversation branching** support.
This feature allows:

- Maintainance of multiple independent branches from any point in a conversation (exactly like Git branches).
- Recovery of a complete chat history for any branch or endpoint.
- Reconstruction and re-sending any version of a conversation history to the LLM.
- Choice of **no history**, **SDK-managed**, **local Postgres history**, or **server-side history** on a per-request basis.

The implementation is built directly on existing partitioned `requests` and `responses` tables with minimal schema extensions (`tree_id`, `branch_id`, `parent_response_id`, `sequence`). No content is duplicated — all message data remains in the original tables.

**Key design principles:**
- Zero duplication of message content.
- Referential integrity via foreign keys.
- Partition-aware and performant (indexes on all branching columns).
- Backward-compatible with all existing data.

## Schema Extensions (Git-style)

- **`conversations` table** – Lightweight metadata per chat tree (`tree_id`, `conversation_id`, `title`, etc.).
- **`requests` & `responses` tables** – Extended with:
  - `tree_id` (root of the entire conversation)
  - `branch_id` (unique linear branch)
  - `parent_response_id` (pointer to the previous response, like a Git parent commit)
  - `sequence` (order within the branch)

## Helper Methods in `xAIPersistenceManager`

The following methods are available on the persistence manager (automatically injected into `XAIClient`):

| Method                                                                                                                                                                                  | Description                                                                                              | Return Type                              |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|------------------------------------------|
| `create_conversation_tree(title: str \| None = None, meta: dict \| None = None)`                                                                                                        | Creates a new conversation tree and returns its identifiers.                                             | `dict` with `tree_id`, `conversation_id` |
| `save_to_history(request_id: UUID, response_id: UUID, tree_id: UUID, branch_id: UUID, parent_response_id: UUID \| None = None, sequence: int \| None = None, role: str, content: dict)` | Links a request/response pair into the history tree. Called automatically by the client.                 | `None`                                   |
| `load_branch_history(tree_id: UUID, branch_id: UUID \| None = None, end_response_id: UUID \| None = None)`                                                                              | Reconstructs the full ordered history for a branch (or the entire tree if `branch_id` is omitted).       | `list[dict]` (ready for `xAIRequest`)    |
| `create_branch(from_response_id: UUID, new_branch_id: UUID \| None = None)`                                                                                                             | Forks a new branch from any existing response.                                                           | `UUID` (the new `branch_id`)             |
| `list_branches(tree_id: UUID)`                                                                                                                                                          | Returns all active branches for a tree with their latest response and metadata.                          | `list[dict]`                             |
| `get_history_for_llm(tree_id: UUID, branch_id: UUID \| None = None)`                                                                                                                    | Convenience method that returns messages in the exact format required by `xAIRequest` / `create_chat()`. | `list[dict[str, Any]]`                   |

All methods are async, fully typed, and integrate with existing structured logger and error hierarchy.

## Quick Start Examples

### 1. Starting a New Conversation (Root Branch)

```python
persistence = xAIPersistenceManager(...)  # with media_root etc.

tree_info = await persistence.create_conversation_tree(title="Research on Grok-4")
tree_id = tree_info["tree_id"]
branch_id = uuid.uuid4()   # first branch

# First turn
response = await client.create_chat(
    messages=[{"role": "user", "content": "Explain prompt caching..."}],
    model="grok-4",
    save_mode="postgres",
    tree_id=tree_id,
    branch_id=branch_id,
)
```

### 2. Creating a Branch (Forking)

```python
# Fork from the last response of the original branch
new_branch_id = await persistence.create_branch(
    from_response_id=response.id,          # or any previous response_id
    new_branch_id=None                     # auto-generates if omitted
)

# Continue on the new branch
response2 = await client.create_chat(
    messages=[{"role": "user", "content": "Now try a different approach..."}],
    model="grok-4",
    tree_id=tree_id,
    branch_id=new_branch_id,
    parent_response_id=response.id,        # optional – enforced automatically
)
```

### 3. Loading & Re-using a Specific Branch History

```python
# Reconstruct full history for a branch
history = await persistence.get_history_for_llm(tree_id, branch_id=new_branch_id)

# Send the entire branch to the LLM (e.g., for a new continuation or comparison)
response_new = await client.create_chat(
    messages=history + [{"role": "user", "content": "Summarise everything so far"}],
    model="grok-4",
    tree_id=tree_id,
    branch_id=new_branch_id,   # continues the same branch
)
```

### 4. Listing All Branches of a Tree

```python
branches = await persistence.list_branches(tree_id)
for b in branches:
    print(f"Branch {b['branch_id']}: {b['message_count']} messages, last at {b['last_tstamp']}")
```

### 5. Traversing the Full Tree (Advanced)

```python
full_history = await persistence.load_branch_history(tree_id)   # all branches
# Returns ordered list per branch – can be rendered as a tree view if needed
```

## Architecture & Integration Points

```
XAIClient (any mode: turn / stream)
   ↓
xAIRequest (now accepts tree_id, branch_id, parent_response_id)
   ↓
create_turn_chat_session / generate_stream_and_persist
   ↓
xAIPersistenceManager
   ├── create_conversation_tree()
   ├── save_to_history()          ← called automatically
   ├── create_branch()
   ├── load_branch_history()
   └── get_history_for_llm()      ← returns ready-to-use messages
```

The client automatically passes `tree_id` / `branch_id` through to persistence when supplied. If omitted, the system falls back to legacy behaviour (`meta.conversation_id` only).

## Persistence Guarantee

- Every request/response pair is still written exactly once.
- Branching metadata is written atomically with the response.
- History reconstruction uses efficient indexed queries (no full table scans).
- Works identically for `mode="turn"` and `mode="stream"`.
- Media files, structured output, and reasoning traces are preserved as before.

## History Modes (Future Extension)

The client will soon expose a `history_mode` parameter:
- `"none"` – no history
- `"sdk"` – let xAI SDK manage state (`previous_response_id`)
- `"local"` – use Postgres branching (default when `tree_id` is supplied)
- `"server"` – combine local branching with xAI server-side continuation

## Further Reading

- `src/ai_api/core/xai/persistence_xai.py` – all branching methods
- `src/ai_api/data_structures/xai_objects.py` – extended `xAIRequest`
- `db_responses_schema.py` – updated schema

---

*This document reflects the refactored architecture (April 2026) with Git-style branching support.*
