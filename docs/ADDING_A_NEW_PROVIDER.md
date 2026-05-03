# Adding a New Provider – Provider-Specific Files Only

## Purpose of This Guide

This README explains **exactly** what directories and files you must create inside `src/ai_api/core/` when adding support for a brand-new LLM provider.

**Strict scope**:  
Only the provider-specific subdirectory and the files that live inside it (`core/<provider>/...`).  
Nothing about `client_factory.py`, `base_provider.py`, `data_structures/`, registration, tests, or any other integration step is covered here.

The canonical example used throughout is the existing **`xai`** provider.

---

## Step 1: Create the Provider Directory

```bash
mkdir -p src/ai_api/core/<provider_name>
```

- Use lowercase with underscores if needed (e.g. `groq`, `anthropic`, `openai_compat`).
- Example for the xAI provider that already exists:

```bash
ls src/ai_api/core/xai/
# chat_batch_xai.py  chat_stream_xai.py  chat_turn_xai.py  errors_xai.py
```

---

## Step 2: Create the Provider-Specific Files

Every new provider needs (at minimum) the four files below. Two additional files are optional and only created when the provider actually supports the feature.

### Required Files (modeled directly on `xai/`)

| File                        | Must Create? | What it contains (xAI example) |
|-----------------------------|--------------|--------------------------------|
| `errors_<provider>.py`     | **Yes**     | Full exception hierarchy + wrapper functions |
| `chat_turn_<provider>.py`  | **Yes**     | Non-streaming chat completion logic |
| `chat_stream_<provider>.py`| **Yes**     | Token-by-token streaming logic + final response assembly |
| `chat_batch_<provider>.py` | Only if the provider offers a batch API | Batch job submission, polling and result retrieval |

### Optional File

| File                          | Create When?                  | Example in repo |
|-------------------------------|-------------------------------|-----------------|
| `embeddings_<provider>.py`   | Provider has an embeddings endpoint | `ollama/embeddings_ollama.py` (xAI currently has none) |

**Resulting directory for a new provider called "groq"**:

```
src/ai_api/core/groq/
├── errors_groq.py
├── chat_turn_groq.py
├── chat_stream_groq.py
├── chat_batch_groq.py     # ← only if Groq batch API is used
└── embeddings_groq.py     # ← only if Groq embeddings are supported
```

---

## File-by-File Details (Using xAI as the Template)

### 1. `errors_<provider>.py`

**Location**: `src/ai_api/core/xai/errors_xai.py`

**Purpose**:  
Define every exception the provider can raise, plus convenience wrapper functions that enrich exceptions with `__cause__` and a `details` dict.

**What the xAI version contains** (you should replicate this structure):

- Imports base classes from `..common.errors`
- Root alias: `xAIError = AIAPIError`
- API-level errors:
  - `xAIAPIError`
  - `xAIAPIConnectionError`
  - `xAIAPIAuthenticationError`
  - `xAIAPIInvalidRequestError`
  - `xAIAPIRateLimitError`
- Client-level errors (internal to the client class):
  - `xAIClientError`
  - `UnsupportedThinkingModeError`
  - `xAIClientBatchError`
  - `xAIClientMultimodalError`
  - `xAIClientCacheError`
- Wrapper functions: `wrap_xai_api_error(exc, message)`
- Full `__all__` export list (including historical backward-compatible names)

**For a new provider** you will typically need at least:
- `<Provider>Error`
- `<Provider>APIError` + connection / auth / rate-limit / invalid-request subclasses
- `<Provider>ClientError`
- `wrap_<provider>_api_error(...)` and `wrap_<provider>_error(...)`

Copy the entire skeleton from `xai/errors_xai.py` and replace names + add/remove subclasses according to the new provider’s actual error surface (gRPC vs HTTP, local vs remote, etc.).

---

### 2. `chat_turn_<provider>.py`

**Location**: `src/ai_api/core/xai/chat_turn_xai.py`

**Purpose**:  
Perform a single, non-streaming chat completion and return a fully populated provider response object.

**Key elements from the xAI implementation**:

- Async function (usually `async def generate_turn(...)`)
- Accepts a provider-specific request object (e.g. `xAIRequest`)
- Instantiates the official SDK client
- Calls the non-streaming chat endpoint
- Converts the raw SDK response into the provider’s `xAIResponse` via `.from_sdk()` or `.from_dict()`
- Handles optional `response_model` (structured JSON output) and attaches `.parsed`
- Returns the `xAIResponse` instance

All SDK calls, request shaping, and response wrapping stay inside this file.

---

### 3. `chat_stream_<provider>.py`

**Location**: `src/ai_api/core/xai/chat_stream_xai.py`

**Purpose**:  
Stream tokens in real time while still producing a complete, persistable final response.

**Key elements from the xAI implementation** (the most involved file):

- Async generator: `async def generate_stream_and_persist(...) -> AsyncIterator[...]`
- Iterates over the SDK’s async streaming iterator
- Immediately `yield`s every raw chunk (for low-latency UX)
- Accumulates text deltas in a list
- On the final chunk (when `is_final` / `finish_reason` appears):
  - Builds a complete `xAIResponse` from the accumulated text + metadata (`raw`, `finish_reason`, `model`, etc.)
  - Optionally validates structured output and sets `.parsed`
- Persists **only the final assembled response** (the original request was already persisted by the caller)

The accumulation + final-assembly pattern must be followed exactly so that persistence and branching work identically for every provider.

---

### 4. `chat_batch_<provider>.py` (conditional)

**Location**: `src/ai_api/core/xai/chat_batch_xai.py`

**Create this file only if** the provider exposes a native batch API (xAI does; most local providers do not).

**What it should contain** (pattern from xAI):
- Functions to submit a batch of requests
- Poll for completion status
- Retrieve and parse results into a ` <Provider>BatchResponse` object
- Error handling specific to batch jobs (`xAIClientBatchError`, etc.)

---

### 5. `embeddings_<provider>.py` (conditional)

**Create this file only if** the provider offers an embeddings endpoint.

**Reference implementation**: `src/ai_api/core/ollama/embeddings_ollama.py`

Typical contents:
- Async function that accepts text(s) + model
- Calls the provider’s embedding endpoint
- Returns a provider-specific embedding response object (or list of vectors)

---

## Naming & Coding Conventions (Must Follow)

| Item                  | Rule (xAI example)                  | Good for new provider          |
|-----------------------|-------------------------------------|--------------------------------|
| Directory             | `xai`                               | `groq`, `anthropic`            |
| Error file            | `errors_xai.py`                     | `errors_groq.py`               |
| Turn file             | `chat_turn_xai.py`                  | `chat_turn_groq.py`            |
| Stream file           | `chat_stream_xai.py`                | `chat_stream_groq.py`          |
| Response class        | `xAIResponse`                       | `GroqResponse`                 |
| Request class         | `xAIRequest`                        | `GroqRequest`                  |
| Error base            | `xAIError`                          | `GroqError`                    |
| Wrapper function      | `wrap_xai_api_error`                | `wrap_groq_api_error`          |
| Main stream function  | `generate_stream_and_persist`       | same name (keep consistent)    |

- All files must be **pure Python** with clear module-level docstrings that include a “High-level view” and “See Also” section (copy style from xAI files).
- Inside these files you may import the provider’s official SDK and the corresponding objects from `data_structures/<provider>_objects.py`.
- Do **not** import or reference anything from `core/client_factory.py` or `core/base_provider.py` in these files.

---

## Quick Checklist for a New Provider “foo”

- [ ] `mkdir -p src/ai_api/core/foo`
- [ ] `errors_foo.py` – full exception hierarchy + wrappers
- [ ] `chat_turn_foo.py` – non-streaming completion
- [ ] `chat_stream_foo.py` – streaming + final-response assembly
- [ ] `chat_batch_foo.py` – **only** if batch API exists
- [ ] `embeddings_foo.py` – **only** if embeddings endpoint exists
- [ ] All names follow the `foo` / `Foo` / `foo_` pattern shown above
- [ ] Docstrings match the style and level of detail used in the `xai` equivalents

---

Once these files exist inside `core/foo/`, the provider is ready for the (separate) integration step that wires it into the rest of the library.

This is the complete, self-contained recipe for the provider-specific layer.