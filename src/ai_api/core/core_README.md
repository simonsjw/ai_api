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

- `create_chat(...)` — unified entry point for **all four modes** (`turn`/`stream`/`batch` use `messages`; `embed` uses `input: str | list[str]`)
- `aclose()` for graceful shutdown
- Provider-specific extras (Ollama: `pull_model()`, `show_model()`, `get_model_options()`; xAI: `list_models()`, `get_model_info()`)

### 2. Modes

All providers support **four** interaction modes via the `mode` parameter (or dedicated client classes such as `TurnOllamaClient`, `EmbedXAIClient`, etc.). The `ChatMode` type is `Literal["turn", "stream", "batch", "embed"]`.

Every mode re-uses the **same** `create_chat(...)` public entry point on the client (for API consistency across providers and modes), but the underlying implementation, return value, and semantics differ.

#### `turn` — Single-turn, non-streaming chat completion
**What it does**: Sends a full conversation history (system/user/assistant/tool messages) and blocks until the model returns the *complete* response. This is the default and most common mode.

**Return type**: A single `OllamaResponse` or `xAIResponse` (contains `.text`, optional `.parsed` Pydantic object when `response_model` is supplied, usage statistics, finish reason, raw payload, etc.).

**Value proposition**: Simple, predictable, full response object available immediately. Perfect for scripts, RAG pipelines, tool-calling loops, and any situation where you need the entire answer before proceeding.

#### `stream` — Real-time token streaming
**What it does**: Identical request to `turn`, but the model streams tokens back as they are generated (async iterator / server-sent events style). The final assembled response is still persisted exactly once.

**Return type**: `AsyncIterator[LLMStreamingChunkProtocol]` — iterate with `async for chunk in client.create_chat(...)`.

**Comparison to `turn`**:
- **Similarities**: Same parameters (`messages`, `model`, `temperature`, `response_model`, `save_mode`, **kwargs), same persistence behaviour, same structured-output support (applied to the final accumulated text).
- **Differences**: `stream` yields progressive output (lower perceived latency for users, great for chat UIs or long generations); `turn` waits for completion (simpler control flow, easier to reason about errors/timeouts, full object ready immediately).
- **When to choose `stream`**: Interactive applications, live demos, or any UI where you want to show "typing" behaviour.
- **When to choose `turn`**: Background jobs, evaluation scripts, or when you need the complete structured object before the next step.

#### `batch` — Multi-conversation processing (simulated or native)
**What it does**: Accepts either a single conversation *or* a list of conversations and executes multiple independent turn-style calls. Ollama simulates batching client-side (sequential by default, or `concurrent=True` via `asyncio.gather`); xAI uses the provider's native batch endpoint for efficiency.

**Return type**: `list[OllamaResponse | xAIResponse]` (or a single response object if you passed only one conversation).

**Comparison to `turn`**:
- Essentially "turn" repeated N times with shared generation parameters.
- Adds `concurrent` flag (Ollama only) and, on xAI, the ability to supply a *list* of `response_model`s (different Pydantic model per conversation).
- **Value proposition**: Efficient bulk processing (dataset evaluation, batch summarisation, synthetic data generation) without writing your own loop or retry logic. Native xAI batch can offer better throughput/cost; simulated Ollama batch gives you the full rich parameter surface (num_ctx, mirostat, think, etc.).

#### `embed` — Text embedding generation (non-conversational)
**What it does**: Generates dense vector embeddings for one or more input strings using the provider's dedicated embedding endpoint (`/api/embed` on Ollama, OpenAI-compatible `/v1/embeddings` on xAI). Embeddings are **never** part of Git-style conversation trees (`branching=False`, `kind="embedding"` in persistence).

**Return type**: `OllamaEmbedResponse` or `XAIEmbedResponse` — contains `.embeddings` (list of float vectors), `.model`, `.usage`, `.raw`, plus `to_neutral_format()` for unified persistence.

**Comparison to chat modes (`turn`/`stream`/`batch`)**:
- Completely different purpose: no message history, no generation, no streaming, no tools.
- Uses the *same* `create_chat(input=..., model=...)` signature on `Embed*Client` instances for uniformity (the `input` parameter replaces `messages`).
- Deliberately excluded from branching so your conversation trees stay clean.
- **Value proposition**: First-class support for RAG, semantic search, clustering, deduplication, or any vector-based workflow. Because embeddings are persisted via the same `PersistenceManager` as chat turns, you can later query "all embedding calls for this model" or join them with chat history in analytics.

**Quick-reference table**

| Mode     | Primary Use Case                     | Return Type                          | Key Extra Feature                  | Branching / Persistence |
|----------|--------------------------------------|--------------------------------------|------------------------------------|-------------------------|
| `turn`   | Single complete answer               | Single Response object               | Full structured output             | Yes (full)              |
| `stream` | Progressive / low-latency UI         | Async iterator of chunks             | Real-time token delivery           | Yes (final only)        |
| `batch`  | Bulk / dataset processing            | list[Response] (or single)           | `concurrent`, per-item models (xAI)| Yes (per item)          |
| `embed`  | Vector search, RAG, similarity       | EmbedResponse (vectors + usage)      | `kind="embedding"`, no branching   | Yes (but non-chat)      |

All four modes support `save_mode` ("none" | "json_files" | "postgres") and forward provider-specific parameters via `**kwargs`. The `Embed*Client` classes are the only ones that accept an `input: str | list[str]` parameter instead of `messages`.

This design gives you **one consistent API surface** while still exposing the unique strengths of each provider and each interaction style.

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

| Feature               | Ollama                                                          | xAI                                               |
|-----------------------|-----------------------------------------------------------------|---------------------------------------------------|
| **Transport**         | Native HTTP (`httpx`)                                           | Official SDK (gRPC + HTTP)                        |
| **Batch**             | Simulated (sequential/concurrent)                               | **Native** + per-request `response_model` lists   |
| **Embeddings**        | First-class (`/api/embed`)                                      | OpenAI-compatible `/v1/embeddings` (delegated via `embeddings_xai.py`) |
| **Model Management**  | `pull_model()`, `show_model()`, `get_model_options()`           | `list_models()`, `get_model_info()`               |
| **Generation Params** | Very rich (`num_ctx`, `repeat_penalty`, `think`, `mirostat`...) | Standard + thinking mode                          |
| **Error Types**       | HTTP + GPU memory warning                                       | Rich (rate limit, auth, multimodal, cache, batch) |
| **Local / Remote**    | Local (localhost:11434)                                         | Remote (api.x.ai)                                 |

## Adding a New Provider

1. Create `core/newprovider/` with `chat_turn_*.py`, `chat_stream_*.py`, `chat_batch_*.py`, `embeddings_*.py` (and `errors_*.py` if needed).
2. Create `core/newprovider_client.py` with `Turn*Client`, `Stream*Client`, `Batch*Client`, `Embed*Client` classes + factory function (`NewProviderClient(...)`).
3. At the bottom of `newprovider_client.py` call:

```python
from .client_factory import register_provider
register_provider("newprovider",
    TurnNewProviderClient, StreamNewProviderClient,
    BatchNewProviderClient, EmbedNewProviderClient)
```

That's it — `get_llm_client("newprovider", mode="embed", ...)` (and all other modes) now works everywhere.

## Provider Module Convention (2026 Architecture)

All providers follow a strict **thin-client + mode-specific modules** pattern.
This ensures:

- The public ``<provider>_client.py`` (e.g. ``ollama_client.py``) contains
  only the four mode classes (``Turn*Client``, ``Stream*Client``,
  ``Batch*Client``, ``Embed*Client``), the factory function, and shared
  base-class lifecycle (``Base*Client``).
- Every mode-specific implementation lives in a sub-package
  ``<provider>/`` (``ollama/`` or ``xai/``):
  - ``chat_turn_<provider>.py`` → ``create_turn_chat_session(...)``
  - ``chat_stream_<provider>.py`` → ``generate_stream_and_persist(...)``
  - ``chat_batch_<provider>.py`` → ``create_batch_chat(...)``
  - ``embeddings_<provider>.py`` → ``create_embeddings(...)``
- This makes adding a new provider or extending a mode (e.g. new batch
  strategy) a localised change with zero impact on the public API surface.

The convention is documented in every module docstring (NumPy style) and
enforced by the ``client_factory.register_provider`` call at the bottom of
each ``*_client.py``.

## Methods and Signatures by Provider Type

All providers expose an identical public surface via ``create_chat`` (the
``LLMProviderAdapter`` protocol). Provider-specific generation parameters
appear only in the Ollama signatures. Return types use the provider's
response class (``OllamaResponse`` / ``xAIResponse`` / ``OllamaEmbedResponse``
/ ``XAIEmbedResponse``).

### Turn Mode – ``Turn<Provider>Client.create_chat``

**Ollama**
::
    async def create_chat(
        self,
        messages: list[dict] | None = None,
        model: str = "llama3.2",
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        repeat_penalty: float | None = None,
        num_ctx: int | None = None,
        stop: list[str] | None = None,
        mirostat: int | None = None,
        think: bool | None = None,
        save_mode: SaveMode = "none",
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> OllamaResponse

**xAI**
::
    async def create_chat(
        self,
        messages: list[dict] | None = None,
        model: str = "grok-4",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        save_mode: SaveMode = "none",
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,          # thinking_mode, top_p, etc.
    ) -> xAIResponse

### Stream Mode – ``Stream<Provider>Client.create_chat``

**Both providers** (identical signature except return chunk type)
::
    async def create_chat(
        self,
        messages: list[dict[str, Any]],
        model: str = "llama3.2" | "grok-4",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        save_mode: SaveMode = "none",
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamingChunkProtocol]

### Batch Mode – ``Batch<Provider>Client.create_chat``

**Ollama** (simulated, supports ``concurrent``)
::
    async def create_chat(
        self,
        messages: list[dict] | list[list[dict]],
        model: str = "llama3.2",
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        repeat_penalty: float | None = None,
        num_ctx: int | None = None,
        stop: list[str] | None = None,
        mirostat: int | None = None,
        think: bool | None = None,
        save_mode: SaveMode = "none",
        concurrent: bool = False,
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> list[OllamaResponse] | OllamaResponse

**xAI** (native, supports per-item ``response_model`` list)
::
    async def create_chat(
        self,
        messages: list[dict] | list[list[dict]],
        model: str = "grok-4",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        save_mode: SaveMode = "none",
        response_model: type[BaseModel] | list[Type[BaseModel]] | None = None,
        concurrent: bool = False,   # accepted for symmetry, ignored
        **kwargs: Any,
    ) -> list[xAIResponse] | xAIResponse

### Embed Mode – ``Embed<Provider>Client.create_chat``

**Ollama**
::
    async def create_chat(
        self,
        input: str | list[str],
        model: str = "nomic-embed-text",
        *,
        save_mode: SaveMode = "none",
        **kwargs: Any,
    ) -> OllamaEmbedResponse

**xAI**
::
    async def create_chat(
        self,
        input: str | list[str],
        model: str = "text-embedding-3-large",
        *,
        save_mode: SaveMode = "none",
        **kwargs: Any,          # dimensions, encoding_format, ...
    ) -> XAIEmbedResponse

### Provider-Specific Convenience Methods (on Base classes)

**Ollama (BaseOllamaClient)**
- ``list_models() -> list[dict]``
- ``pull_model(name: str, stream: bool = False) -> dict | AsyncIterator[dict]``
- ``show_model(name: str) -> dict``
- ``get_model_options(name: str) -> dict``

**xAI (BaseXAIClient)**
- ``list_models() -> list[dict]``
- ``get_model_info(model: str) -> dict``

All methods are fully documented with NumPy-style docstrings in their
respective modules.

---

**`ai_api.core`** provides a production-ready, extensible and consistent interface to any LLM.
