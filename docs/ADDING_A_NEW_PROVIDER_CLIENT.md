# Adding a New Provider – The Client Entry Point (`<provider>_client.py`)

## 1. Why This File Is the "Front Door"

This is the **single integration point** that makes a new provider appear in the public API of `ai_api`.

After you have created:
- the data structures (`<provider>_objects.py`) — see `ADDING_A_NEW_PROVIDER_DATA_STRUCTURES.md`
- the provider-specific logic (`core/<provider>/chat_*.py`, `errors_*.py`) — see `ADDING_A_NEW_PROVIDER.md`

…you only need to create one more file:

```
src/ai_api/core/<provider>_client.py
```

This file:
- Defines four thin client classes (one per mode)
- Provides a convenient `<Provider>Client(...)` factory function
- Registers the provider with the central `client_factory`

Once registered, users can do:

```python
from ai_api.core.client_factory import get_llm_client
client = get_llm_client("groq", logger=logger, mode="stream", api_key=...)
```

---

## 2. How It Relates to `base_provider.py`

`base_provider.py` defines a single structural protocol:

```python
@runtime_checkable
class LLMProviderAdapter(Protocol):
    async def create_chat(self, *args: Any, **kwargs: Any) -> Any: ...
```

Every mode-specific client class you write must implement a method called `create_chat`.  
Because the protocol is structural (`@runtime_checkable`), **no inheritance is required** — the factory and higher-level code simply check that the method exists with a compatible signature.

The `create_chat` method is the **unified entry point**. It hides whether the caller wants turn-based, streaming, batch, or embeddings behind a single async method.

---

## 3. The Registry Pattern in `client_factory.py`

The factory maintains a simple registry:

```python
PROVIDER_REGISTRY: dict[str, dict[ChatMode, Type]] = {}
```

A provider registers itself by calling (usually at the bottom of its `*_client.py`):

```python
register_provider(
    "groq",
    GroqTurnClient,
    GroqStreamClient,
    GroqBatchClient,
    GroqEmbedClient,   # or None if no embeddings
)
```

`get_llm_client()` then looks up the right class for the requested `mode` and instantiates it.

**Lazy auto-registration** (built-in for ollama/xai, and we will add it for your new provider) means users never have to manually import the client module.

---

## 4. What You Must Write in `<provider>_client.py`

### 4.1 Class Hierarchy (recommended pattern from xAI)

```python
class BaseGroqClient:
    """Shared initialisation, HTTP client, SDK client, persistence, etc."""
    def __init__(self, logger, api_key, base_url=..., timeout=120,
                 persistence_manager=None, **kwargs):
        self.logger = logger
        self.api_key = api_key
        ...
        self.persistence_manager = persistence_manager

class GroqTurnClient(BaseGroqClient):
    async def create_chat(self, **kwargs):
        from .groq.chat_turn_groq import generate_turn   # or create_turn_chat_session
        return await generate_turn(..., persistence_manager=self.persistence_manager)

class GroqStreamClient(BaseGroqClient):
    async def create_chat(self, **kwargs):
        from .groq.chat_stream_groq import generate_stream_and_persist
        return generate_stream_and_persist(...)   # returns async iterator

class GroqBatchClient(BaseGroqClient):
    async def create_chat(self, **kwargs):
        from .groq.chat_batch_groq import create_batch_chat
        return await create_batch_chat(...)

class GroqEmbedClient(BaseGroqClient):
    async def create_chat(self, **kwargs):
        # call /embeddings endpoint directly or via SDK
        ...
```

### 4.2 Convenience Factory Function

```python
def GroqClient(logger, mode: ChatMode = "turn", **kwargs):
    if mode == "turn":   return GroqTurnClient(logger=logger, **kwargs)
    if mode == "stream": return GroqStreamClient(logger=logger, **kwargs)
    if mode == "batch":  return GroqBatchClient(logger=logger, **kwargs)
    if mode == "embed":  return GroqEmbedClient(logger=logger, **kwargs)
    raise ValueError(f"Unsupported mode: {mode}")
```

### 4.3 Registration (the one-line magic)

At the very bottom of the file:

```python
from .client_factory import register_provider

register_provider(
    "groq",
    GroqTurnClient,
    GroqStreamClient,
    GroqBatchClient,
    GroqEmbedClient,   # or None
)
```

---

## 5. Integration Changes Required

### 5.1 (Recommended) Add lazy import in `client_factory.py`

Inside `get_llm_client()`, extend the lazy-import block:

```python
if provider not in PROVIDER_REGISTRY:
    if provider == "ollama":
        from . import ollama_client
    elif provider == "xai":
        from . import xai_client
    elif provider == "groq":          # <-- add this line for your new provider
        from . import groq_client
    else:
        raise ValueError(...)
```

This single line gives users the same seamless experience as the built-in providers.

### 5.2 No other changes needed

- `base_provider.py` requires nothing.
- The `core/<provider>/` modules are imported on demand inside the client classes.
- Data structures are imported only where needed.

---

## 6. Complete End-to-End Process (3 Steps)

1. **Data Structures**  
   Create `src/ai_api/data_structures/<provider>_objects.py`  
   → Follow `ADDING_A_NEW_PROVIDER_DATA_STRUCTURES.md`

2. **Provider Logic**  
   Create `src/ai_api/core/<provider>/` with `chat_turn_*.py`, `chat_stream_*.py`, `errors_*.py`, etc.  
   → Follow `ADDING_A_NEW_PROVIDER.md`

3. **Client Entry Point** (this file)  
   Create `src/ai_api/core/<provider>_client.py`  
   → Implement the four mode clients + register_provider call  
   → (Optional but recommended) add the lazy-import line in `client_factory.py`

After step 3 the provider is fully usable via `get_llm_client("groq", ...)`.

---

## 7. Minimal Skeleton for a New Provider ("groq_client.py")

```python
"""
Groq client – thin orchestrator that satisfies LLMProviderAdapter.
Heavy lifting lives in core/groq/chat_*.py and data_structures/groq_objects.py.
"""

import logging
from typing import Any, Optional

from .base_provider import LLMProviderAdapter
from ..data_structures.groq_objects import GroqRequest, GroqResponse, GroqStreamingChunk
from .client_factory import register_provider


class BaseGroqClient:
    def __init__(self, logger: logging.Logger, api_key: str,
                 base_url: str = "https://api.groq.com/openai/v1",
                 timeout: int = 120,
                 persistence_manager: Any | None = None,
                 **kwargs):
        self.logger = logger
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.persistence_manager = persistence_manager
        # initialise httpx or SDK client here


class GroqTurnClient(BaseGroqClient):
    async def create_chat(self, **kwargs) -> GroqResponse:
        from .groq.chat_turn_groq import generate_turn
        return await generate_turn(
            request=GroqRequest(**kwargs),
            persistence_manager=self.persistence_manager,
            logger=self.logger,
        )


class GroqStreamClient(BaseGroqClient):
    async def create_chat(self, **kwargs):
        from .groq.chat_stream_groq import generate_stream_and_persist
        return generate_stream_and_persist(
            request=GroqRequest(**kwargs),
            persistence_manager=self.persistence_manager,
            logger=self.logger,
        )


class GroqBatchClient(BaseGroqClient):
    async def create_chat(self, **kwargs):
        from .groq.chat_batch_groq import create_batch_chat
        return await create_batch_chat(...)


class GroqEmbedClient(BaseGroqClient):
    async def create_chat(self, **kwargs):
        # embeddings implementation
        ...


def GroqClient(logger: logging.Logger, mode: str = "turn", **kwargs):
    if mode == "turn":   return GroqTurnClient(logger=logger, **kwargs)
    if mode == "stream": return GroqStreamClient(logger=logger, **kwargs)
    if mode == "batch":  return GroqBatchClient(logger=logger, **kwargs)
    if mode == "embed":  return GroqEmbedClient(logger=logger, **kwargs)
    raise ValueError(f"Unknown mode {mode}")


# === REGISTRATION (must be last) ===
register_provider("groq", GroqTurnClient, GroqStreamClient, GroqBatchClient, GroqEmbedClient)
```

---

## 8. Checklist & Common Gotchas

- [ ] All four mode clients implement `async def create_chat(...)`
- [ ] `register_provider(...)` is called exactly once at module import time
- [ ] Lazy-import line added in `client_factory.py` (so `get_llm_client("groq")` works out of the box)
- [ ] Client classes accept the common kwargs (`logger`, `persistence_manager`, `timeout`, provider-specific keys)
- [ ] Heavy logic lives in `core/<provider>/chat_*.py` — keep this file thin
- [ ] Import data structures only where actually used (avoid circular imports)
- [ ] Test with `get_llm_client("groq", logger=..., mode="turn")` immediately after registration

**Common pitfall**: forgetting the `register_provider` call — the provider will silently be missing from the registry.

---

This file completes the provider integration. Using this with the two previous guides, provides a complete, repeatable recipe for adding any new LLM provider to `ai_api`.
