# `ai_api.data_structures` — Shared Data Models & Protocols

The `data_structures` package is the **foundational layer** of `ai_api`. It defines the contracts and concrete models that allow the entire library to remain provider-agnostic while still supporting rich, provider-specific functionality.

## Purpose

- Provide **minimal, stable protocols** that every request, response, and streaming chunk must implement.
- Supply **provider-specific concrete implementations** for Ollama and xAI.
- Centralize the `SaveMode` type and structured-output specification classes.
- Enable the powerful symmetrical persistence system in `core/common/persistence.py`.

Everything in `core/` (clients, factory, chat modules) ultimately depends on the types defined here.

## Directory Structure

```
    data_structures/
    ├── base_objects.py          # Core protocols + LLMEndpoint + SaveMode (the heart of the system)
    ├── ollama_objects.py        # Ollama-specific request/response/streaming models + OllamaJSONResponseSpec
    ├── xai_objects.py           # xAI-specific request/response/streaming models + xAIJSONResponseSpec
    └── db_responses_schema.py   # Database table schemas for persistence (Postgres)
```

## Core Concepts

### 1. Protocols (in `base_objects.py`)

These are the **single most important** abstractions in the entire library:

| Protocol                    | Required Methods / Properties                 | Used By                                 |
|-----------------------------|-----------------------------------------------|-----------------------------------------|
| `LLMRequestProtocol`        | `meta()`, `payload()`, `endpoint()`           | `PersistenceManager.persist_request()`  |
| `LLMResponseProtocol`       | `meta()`, `payload()`, `endpoint()`           | `PersistenceManager.persist_response()` |
| `LLMStreamingChunkProtocol` | `text`, `finish_reason`, `is_final`, `raw`... | Streaming clients                       |
| `LLMEndpoint`               | `provider`, `model`, `base_url`, ...          | Both request & response                 |

Any object that implements these methods (via duck typing) can be persisted, logged, or passed around without the rest of the code knowing whether it came from Ollama or xAI.

### 2. Provider-Specific Objects

- **`ollama_objects.py`** — Contains `OllamaRequest`, `OllamaResponse`, `OllamaStreamingChunk`, `OllamaEmbedRequest/Response`, and `OllamaJSONResponseSpec`.
- **`xai_objects.py`** — Contains the equivalent `xAIRequest`, `xAIResponse`, `xAIStreamingChunk`, and `xAIJSONResponseSpec`.

These classes implement the protocols above **and** add provider-specific fields and helper methods (`to_ollama_dict()`, `to_sdk_chat_kwargs()`, `from_dict()`, etc.).

### 3. Structured Output Specifications

- `OllamaJSONResponseSpec` and `xAIJSONResponseSpec` are created by `core/common/response_struct.create_json_response_spec()`.
- They convert a Pydantic model (or raw schema) into the exact format each provider expects.

### 4. `SaveMode`

```python
    type SaveMode = Literal["none", "json_files", "postgres"]
```

Defined centrally in `base_objects.py` and used everywhere persistence is involved.

## How It Fits Into the Bigger Picture

```
    User Code / Clients (core/ollama_client.py, core/xai_client.py)
              │
              ▼
    Provider Modules (core/ollama/, core/xai/)
              │
              ▼
    data_structures/          ← You are here
       ├── Protocols (base_objects.py)
       ├── Ollama models (ollama_objects.py)
       └── xAI models (xai_objects.py)
              │
              ▼
    core/common/persistence.py   ← Uses the protocols for symmetrical storage
```

This separation is what makes the library extensible: to add a new provider the implemtantation of three protocols is needed —  persistence, logging, or the client factory work without any additional adjustments.

## Key Files

| File                     | Role                                                                               |
|--------------------------|------------------------------------------------------------------------------------|
| `base_objects.py`        | The single source of truth for all protocols and `SaveMode`                        |
| `ollama_objects.py`      | All Ollama data models + structured output spec                                    |
| `xai_objects.py`         | All xAI data models + structured output spec                                       |
| `db_responses_schema.py` | SQLAlchemy / raw SQL definitions for the `llm_requests` and `llm_responses` tables |

## Usage Example

```python
    from ai_api.data_structures.base_objects import LLMRequestProtocol, SaveMode
    from ai_api.data_structures.ollama_objects import OllamaRequest

    # Any object that satisfies LLMRequestProtocol can be persisted
    request: LLMRequestProtocol = OllamaRequest(
        model="llama3.2",
        input=[{"role": "user", "content": "Hello"}],
        save_mode="postgres"
    )

    # The persistence layer only cares about the protocol methods
    rid, ts = await persistence_manager.persist_request(request)
```

