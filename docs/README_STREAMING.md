# Streaming Functionality – xAIClient

## Overview

The `mode="stream"` path in `XAIClient` uses the **official xAI Python SDK** (`xai_sdk`).
This is the recommended and maintained streaming implementation.

Benefits:
- Official support for all xAI features (reasoning, tools, caching, etc.)
- Real-time token streaming via the clean `chat.stream()` API
- Single-final-row persistence (request before + complete response after)
- Consistent `LLMStreamingChunkProtocol` interface

## Quick Start

See the usage example in the main README or in the code docstrings.

## Architecture
XAIClient(mode="stream")
└─ StreamXAIClient.create_chat()
├─ xAIRequest (domain model)
├─ xai_sdk.AsyncClient.chat.create(...)
└─ generate_stream_and_persist() [stream_xai.py]
├─ xai_stream() → official .stream()
└─ persistence_manager.persist_* (single final row)
textKey files:
- `src/ai_api/core/xai_client.py` – factory + `StreamXAIClient`
- `src/ai_api/core/xai/stream_xai.py` – SDK orchestration + persistence
- `src/ai_api/data_structures/xai_objects.py` – `LLMStreamingChunkProtocol`, `xAIRequest`, etc.
- `src/ai_api/core/xai/persistence_xai.py` – `xAIPersistenceManager` (your infopypg integration)

## Persistence Guarantee

When `save_mode="postgres"`:
- Request row is written **before** the API call.
- Only **one** response row is written **after** the stream completes.
- No per-chunk writes; database load remains constant regardless of response length.

## Logging

All streaming events are logged via the standard logger (see `logging` library integration).

## Further customisation

For questions or further customisation, refer to the top-level `README.md` or the persistence/logging library repositories.
These updates fully resolve the original inconsistencies (including the linter error around
