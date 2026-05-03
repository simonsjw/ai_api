# Adding a New Provider – Data Structures (`<provider>_objects.py`)

## 1. Philosophy & the Neutral Model

The `ai_api` library is built on a **provider-agnostic core**. Every LLM interaction — whether it comes from Ollama (local), xAI (remote), Groq, Anthropic, or any future provider — is ultimately converted into a single, canonical **neutral format** before being persisted or used for branching.

### Why this design makes sense

- **Git-style branching & history editing** work identically for every provider. The `ChatSession`, `PersistenceManager`, and recursive CTE queries in Postgres only ever see `NeutralTurn` and `NeutralResponseBlob` objects.
- **Zero duplication of conversation history**. Instead of storing raw xAI or Ollama payloads forever, we store one clean `NeutralTurn` per turn. The original provider-specific `raw` dict is kept only for debugging/audit inside the `raw` field.
- **Future-proofing**. Adding a new provider never requires changes to persistence, logging, or the branching engine.
- **Structured output & media** are normalised once, so higher-level code never needs `if provider == "xai"` branches.

**The contract is simple**:  
Every provider-specific response object **must** implement `to_neutral_format(branch_info) -> dict` so that a `NeutralTurn(**result)` can be constructed and stored in the `response` JSONB column.

---

## 2. The Base Layer (`base_objects.py`)

All provider objects live in `src/ai_api/data_structures/` and are built on top of the shared foundation defined in `base_objects.py`.

### Core Protocols (structural typing + runtime checks)

```python
@runtime_checkable
class LLMRequestProtocol(Protocol):
    def endpoint(self) -> LLMEndpoint: ...
    def meta(self) -> dict[str, Any]: ...
    def payload(self) -> dict[str, Any]: ...

class LLMResponseProtocol(Protocol):
    def endpoint(self) -> LLMEndpoint: ...
    def meta(self) -> dict[str, Any]: ...
    def payload(self) -> dict[str, Any]: ...
    def to_neutral_format(self, branch_info: dict | None = None) -> dict: ...
```

### Neutral Models (the single source of truth for persistence)

- **`NeutralTurn`**: The canonical representation of one assistant (or tool) message.
  - `role`, `content` (str | list[dict] for multimodal), `structured`, `tools`, `finish_reason`, `usage`, `raw`, `timestamp`, `branch_meta`
- **`NeutralPrompt`**: The prompt that produced the turn (`system`, `user`, `structured_spec`, `tools`).
- **`NeutralResponseBlob`**: What actually gets written to Postgres:
  ```python
  prompt: NeutralPrompt
  response: NeutralTurn
  branch_context: dict
  ```

### Supporting Types

- `LLMEndpoint` – frozen dataclass describing provider, model, base_url, path, api_type.
- `SaveMode` – Literal["none", "json_files", "postgres"]

---

## 3. What Classes a `<provider>_objects.py` Must Provide

| Feature              | Required Classes                              | Optional / Recommended                  | Notes |
|----------------------|-----------------------------------------------|-----------------------------------------|-------|
| **Turn-based Chat**  | `<Provider>Request`, `<Provider>Response`    | —                                       | Core contract |
| **Streaming Chat**   | `<Provider>Response` (re-used)                | `<Provider>StreamingChunk`              | Strongly recommended for real-time UX |
| **Media / Multimodal** | Handled inside Request & Response            | `has_media()` helper on Request         | Use `content: list[dict]` with `type: "input_text" \| "input_image"` |
| **Embeddings**       | `<Provider>EmbedRequest`, `<Provider>EmbedResponse` | —                                  | Only if the provider exposes `/embeddings` |
| **Batch**            | `<Provider>BatchRequest`, `<Provider>BatchResponse` | —                                | Only if the provider has a native batch API (xAI does) |
| **Structured Output**| `<Provider>JSONResponseSpec`                  | —                                       | Mirrors `xAIJSONResponseSpec` / `OllamaJSONResponseSpec` |

**Example directory entry for a new provider “groq”**:

```
src/ai_api/data_structures/
└── groq_objects.py          # contains GroqRequest, GroqResponse, GroqStreamingChunk, GroqJSONResponseSpec, ...
```

---

## 4. The Canonical Methods Contract

### Methods you **must** implement (Protocol requirements)

| Method                  | On Request? | On Response? | Purpose |
|-------------------------|-------------|--------------|---------|
| `endpoint()`            | Yes         | Yes          | Returns `LLMEndpoint(provider=..., model=...)` |
| `meta()`                | Yes         | Yes          | Generation settings + usage/telemetry |
| `payload()`             | Yes         | Yes          | The actual input/output content |
| `to_neutral_format(...)`| No          | **Yes**      | Converts to `NeutralTurn` dict – **the most important method** |

### Recommended / Commonly Used Helper Methods (not in Protocol but expected by `core/<provider>/chat_*.py`)

- `from_dict(cls, data)` – reconstruct from raw dict / DB
- `from_sdk(cls, sdk_response)` – convert official SDK object (xAI, Ollama, etc.)
- `set_parsed(self, parsed: BaseModel | dict)` – attach structured output (uses `object.__setattr__` because objects are frozen)
- `to_sdk_chat_kwargs(self)` – produce the exact dict the provider’s SDK expects
- `from_neutral_history(cls, history, new_prompt, meta)` – rebuild a request from previous neutral turns (used by branching)

---

## 5. Step-by-Step: Creating `<provider>_objects.py`

### 5.1 Skeleton (copy from xAI and adapt)

```python
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field
from .base_objects import (
    LLMEndpoint, LLMRequestProtocol, LLMResponseProtocol,
    NeutralTurn, NeutralPrompt, NeutralResponseBlob
)

__all__ = ["<Provider>Request", "<Provider>Response", "<Provider>StreamingChunk", ...]

# ------------------------------------------------------------------
# Structured Output Spec (if supported)
# ------------------------------------------------------------------
@dataclass(frozen=True)
class <Provider>JSONResponseSpec(BaseModel):
    model: type[BaseModel] | dict | None = None
    # ... to_sdk_response_format(), parse_json(), etc.
```

### 5.2 The Request Class (example from `xai_objects.py`)

```python
@dataclass(frozen=True)
class xAIRequest(BaseModel, LLMRequestProtocol):
    model: str
    input: str | xAIInput
    temperature: float = 0.7
    max_tokens: int | None = None
    response_format: xAIJSONResponseSpec | None = None
    tools: list[dict] | None = None
    # ... multimodal, cache, etc.

    def endpoint(self) -> LLMEndpoint: ...
    def meta(self) -> dict[str, Any]: ...
    def payload(self) -> dict[str, Any]: ...
    def to_sdk_chat_kwargs(self) -> dict[str, Any]: ...
    @classmethod
    def from_neutral_history(cls, history, new_prompt, meta) -> Self: ...
```

### 5.3 The Response Class (the most important – must implement `to_neutral_format`)

```python
@dataclass(frozen=True)
class xAIResponse(BaseModel, LLMResponseProtocol):
    model: str
    created_at: str | None = None
    choices: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    parsed: BaseModel | dict | None = None   # populated by response_struct

    @property
    def text(self) -> str: ...
    @property
    def tool_calls(self) -> list[dict]: ...

    def endpoint(self) -> LLMEndpoint: ...
    def meta(self) -> dict[str, Any]: ...
    def payload(self) -> dict[str, Any]: ...

    def to_neutral_format(self, branch_info: dict | None = None) -> dict:
        return {
            "role": "assistant",
            "content": self.text,
            "structured": self.parsed,
            "finish_reason": self.choices[0].get("finish_reason") if self.choices else None,
            "usage": self.usage,
            "tools": self.tool_calls,
            "raw": self.raw,
            "timestamp": datetime.utcnow().isoformat(),
            "branch_meta": branch_info or {},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "xAIResponse": ...
    @classmethod
    def from_sdk(cls, sdk_response: Any) -> "xAIResponse": ...
    def set_parsed(self, parsed: BaseModel | dict) -> None:
        object.__setattr__(self, "parsed", parsed)   # frozen dataclass
```

### 5.4 Streaming Chunk (recommended)

```python
@dataclass(frozen=True)
class xAIStreamingChunk:
    text: str = ""
    finish_reason: str | None = None
    is_final: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
    # ... tool_calls_delta, etc.
```

---

## 6. Media / Multimodal & Structured Output

### Media support
- Requests accept `content: str | list[dict[str, Any]]` where each dict has `"type": "input_text" | "input_image" | "input_file"`.
- `xAIRequest.has_media()` and `xAIMessage` helper classes make this ergonomic.
- In neutral format the `content` field simply carries the same list (or a string for text-only).

### Structured / JSON output
- Create a `<Provider>JSONResponseSpec` that knows how to turn a Pydantic model into the provider’s native `response_format` / `format` parameter.
- After generation, `response_struct.py` calls `model_validate_json()` and then `response.set_parsed(parsed)`.
- The neutral `structured` field stores the validated Pydantic instance (or dict).

---

## 7. Best Practices, Patterns & Common Pitfalls

### Patterns to copy
- Use `@dataclass(frozen=True)` + `BaseModel` for immutability + Pydantic validation + JSON serialisation.
- Always implement `set_parsed` with `object.__setattr__` (frozen objects reject normal assignment).
- Keep `raw` as the original SDK payload for debugging.
- Export everything needed via `__all__`.

### Common pitfalls
- **Missing `to_neutral_format`** → Pyrefly error + runtime `AttributeError` when `persist_chat_turn` is called (exactly the error we saw on `xAIResponse`).
- Forgetting `from_sdk()` or `from_dict()` → chat modules cannot construct responses.
- Incorrect neutral keys → branching / reconstruction breaks silently.
- Not handling `parsed` on frozen instances → `FrozenInstanceError`.
- Omitting `endpoint()` → `LLMEndpoint` is required by persistence for routing and logging.

---

## 8. Verification Checklist

- [ ] All four Protocol methods implemented on Request and Response
- [ ] `to_neutral_format()` produces a dict that can be passed to `NeutralTurn(**d)`
- [ ] `from_sdk()` / `from_dict()` round-trip correctly with real SDK responses
- [ ] Media content list format works for both text-only and image+text cases
- [ ] Structured output path (`JSONResponseSpec` + `set_parsed`) functions end-to-end
- [ ] `python -c "from data_structures.<provider>_objects import *; ..."` imports cleanly
- [ ] `to_neutral_format()` output matches the shape expected by `NeutralResponseBlob`

---

Once ` <provider>_objects.py ` exists and satisfies the above, the provider can be used by the corresponding `core/<provider>/chat_*.py` modules with **zero changes** to persistence, branching, or the neutral layer.

This completes the data-structures side of adding a new provider. The companion guide `ADDING_A_NEW_PROVIDER.md` covers the `core/<provider>/` implementation files.