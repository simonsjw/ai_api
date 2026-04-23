# Ollama Embeddings Client (`EmbedOllamaClient`)

**Local, private, high-performance vector embeddings for RAG and semantic search.**

This module adds first-class support for Ollama’s native `/api/embed` endpoint to the `ai_api` package. It is fully consistent with the existing `OllamaClient` architecture (chat, persistence, logging, error handling, and `BaseOllamaClient`).

## Features

- **Full `/api/embed` support** — batch input, `truncate`, `keep_alive`, `dimensions`, model-specific `options`
- **Rich telemetry** — `total_duration`, `load_duration`, `prompt_eval_count`, etc. (nanoseconds)
- **NumPy & SciPy integration** (optional, lazy-loaded)
  - `.to_numpy()` → `np.ndarray` shape `(n_inputs, embedding_dim)`
  - `.cosine_similarity(idx1, idx2)`
- **Optional persistence** — reuse the existing `PersistenceManager` and `SaveMode`
- **Lightweight** — no new hard dependencies
- **Production-ready** — async, proper error wrapping, logging, type hints

---

## Quick Start

```python
import logging
from src.ai_api.core.ollama_client import EmbedOllamaClient
from src.ai_api.core.common.persistence import PersistenceManager  # optional

logger = logging.getLogger(__name__)

# Basic usage (no persistence)
client = EmbedOllamaClient(logger=logger)

resp = await client.create_embeddings(
    model="nomic-embed-text",
    input=["RAG with local embeddings is powerful", "Ollama keeps your data private"],
)

print(resp.n_inputs)        # 2
print(resp.embedding_dim)   # 768 (for nomic-embed-text)
print(len(resp.embeddings[0]))  # 768

# Convert to NumPy
vectors = resp.to_numpy()   # shape: (2, 768), dtype=float32
```

---

## Installation

No extra dependencies required for core functionality.

**Optional (recommended for production RAG):**

```bash
pip install numpy scipy
```

These are **lazily imported** only when either `.to_numpy()` or `.cosine_similarity()` are called.

---

## API Reference & Examples

### 1. Basic Single Embedding

```python
resp = await client.create_embeddings(
    model="mxbai-embed-large",
    input="What is the capital of France?",
)
vector = resp.embeddings[0]          # list of 1024 floats
```

### 2. Batch Embeddings (Recommended for RAG)

```python
documents = [
    "Ollama provides local LLMs and embeddings.",
    "Embeddings enable semantic search without sending data to the cloud.",
    "nomic-embed-text is fast and accurate for general use.",
]

resp = await client.create_embeddings(
    model="nomic-embed-text",
    input=documents,           # list[str] → batch mode
    keep_alive="30m",          # keep model in memory for 30 minutes
)

print(f"Generated {resp.n_inputs} vectors of dimension {resp.embedding_dim}")
# Generated 3 vectors of dimension 768
```

### 3. Using NumPy (Shape Documentation)

All documentation uses **NumPy notation**:

```python
resp = await client.create_embeddings(model="nomic-embed-text", input=docs)

# embeddings: list[list[float]]   → equivalent to np.ndarray shape (n_inputs, embedding_dim)
arr: np.ndarray = resp.to_numpy()          # shape (3, 768), dtype=float32

print(arr.shape)        # (3, 768)
print(arr.dtype)        # float32
print(arr.mean(axis=0)) # mean embedding (useful for centroid)
```

### 4. Cosine Similarity (SciPy Notation)

```python
resp = await client.create_embeddings(
    model="nomic-embed-text",
    input=["cat", "dog", "automobile"]
)

sim_cat_dog = resp.cosine_similarity(0, 1)      # ~0.72 (high similarity)
sim_cat_car = resp.cosine_similarity(0, 2)      # ~0.15 (low similarity)

print(f"cat ↔ dog similarity: {sim_cat_dog:.3f}")
```

**Implementation note**: Uses `scipy.spatial.distance.cosine` under the hood (1 − cosine distance).

### 5. All Parameters

```python
resp = await client.create_embeddings(
    model="snowflake-arctic-embed-m",
    input=["text 1", "text 2", "text 3"],
    truncate=True,                    # default
    keep_alive=300,                   # int seconds or "5m"
    dimensions=512,                   # reduce output dimension (if model supports)
    options={"temperature": 0.0},     # rarely needed for embeddings
)
```

### 6. Persistence Integration (Same as Chat)

```python
from src.ai_api.core.common.persistence import PersistenceManager
from pathlib import Path

pm = PersistenceManager(
    logger=logger,
    db_url="postgresql://...",
    media_root=Path("./media"),   # optional
)

client = EmbedOllamaClient(
    logger=logger,
    persistence_manager=pm,
)

resp = await client.create_embeddings(
    model="nomic-embed-text",
    input=documents,
    save_mode="postgres",         # logs request + response to DB
)
```

The same `SaveMode` (`"none"`, `"postgres"`) and persistence patterns used for chat are reused here.

### 7. Error Handling

```python
try:
    resp = await client.create_embeddings(model="nonexistent-model", input="test")
except Exception as e:
    logger.error(f"Embedding failed: {e}")
    # Automatically wrapped with Ollama-specific context
```

All errors go through `wrap_ollama_error` for consistent handling.

---

## Performance & Telemetry

Ollama returns rich local metrics:

```python
resp = await client.create_embeddings(...)

print(f"Total time: {resp.total_duration / 1e6:.2f} ms")
print(f"Model load: {resp.load_duration / 1e6:.2f} ms")
print(f"Tokens evaluated: {resp.prompt_eval_count}")
```

Typical numbers on a modern GPU:
- `nomic-embed-text` (768 dim): ~5–15 ms per document (batch of 32)
- `mxbai-embed-large` (1024 dim): slightly slower but higher quality

---

## Recommended Embedding Models (2026)

| Model                    | Dimensions | Speed     | Quality (MTEB) | Best For                  |
|--------------------------|------------|-----------|----------------|---------------------------|
| `nomic-embed-text`       | 768        | Very Fast | Excellent      | General RAG, default      |
| `mxbai-embed-large`      | 1024       | Fast      | Top-tier       | High accuracy             |
| `snowflake-arctic-embed-m` | 768      | Fast      | Very good      | Multilingual              |
| `all-minilm`             | 384        | Fastest   | Good           | Resource-constrained      |

Pull with:
```bash
ollama pull nomic-embed-text
```

---

## Full RAG Pipeline Example

```python
# 1. Embed documents
docs = ["doc1 content...", "doc2 content..."]
embed_resp = await embed_client.create_embeddings(
    model="nomic-embed-text",
    input=docs,
    keep_alive="1h"
)
doc_vectors = embed_resp.to_numpy()

# 2. Embed query
query_resp = await embed_client.create_embeddings(
    model="nomic-embed-text",
    input="What is the best local embedding model?"
)
query_vec = query_resp.to_numpy()[0]

# 3. Compute similarities (NumPy)
import numpy as np
similarities = np.dot(doc_vectors, query_vec) / (
    np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vec)
)
top_idx = np.argmax(similarities)
print(f"Most relevant doc: {docs[top_idx]} (score: {similarities[top_idx]:.3f})")
```

---

## Integration with Existing `OllamaClient`

You can also expose embeddings directly on the main client if desired:

```python
# In ollama_client.py (optional extension)
class OllamaClient(...):
    async def embed(self, *args, **kwargs):
        embed_client = EmbedOllamaClient(...)  # or share http client
        return await embed_client.create_embeddings(*args, **kwargs)
```

---

## Summary

| Method / Attribute             | Description             | NumPy/SciPy Equivalent              |
|--------------------------------|-------------------------|-------------------------------------|
| `create_embeddings(...)`       | Main entry point        | —                                   |
| `.embeddings`                  | Raw list of vectors     | `list[list[float]]`                 |
| `.to_numpy()`                  | Convert to array        | `np.ndarray` shape `(n, dim)`       |
| `.cosine_similarity(i, j)`     | Pairwise similarity     | `1 - scipy.spatial.distance.cosine` |
| `.n_inputs` / `.embedding_dim` | Convenience properties  | `.shape[0]` / `.shape[1]`           |
| `total_duration` etc.          | Telemetry (nanoseconds) | —                                   |
