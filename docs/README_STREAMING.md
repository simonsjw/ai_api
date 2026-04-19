# Streaming Functionality – xAIClient

## Overview

The `mode="stream"` path in `XAIClient` uses the **official xAI Python SDK** (`xai_sdk.AsyncClient`).
This is the recommended and actively maintained streaming implementation.

**Key benefits:**
- Real-time token streaming via the clean `chat.stream()` API.
- Full support for all xAI features (reasoning traces, tools, prompt caching, structured output, etc.).
- Unified multimodal handling identical to turn-based mode.
- Single-final-row persistence (request before the call + complete response after the stream finishes).
- Consistent `LLMStreamingChunkProtocol` interface across providers.

## Quick Start

```python
from ai_api import XAIClient
from pathlib import Path

client = XAIClient(
    logger=logger,
    api_key="xai-...",
    mode="stream",
    persistence_manager=xAIPersistenceManager(
        pg_resolved_settings=..., 
        media_root=Path("/path/to/media")   # ← enables automatic media saving
    )
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Describe this image and summarise the attached document."},
            {"type": "input_image", "image_url": "https://example.com/image.jpg"},      # remote URL
            {"type": "input_image", "image_url": "/absolute/path/to/local/photo.png"}, # local file
            {"type": "input_file",  "file_url": "/absolute/path/to/report.pdf"}         # local file
        ]
    }
]

async for chunk in client.create_chat(
    messages=messages,
    model="grok-4",
    temperature=0.7,
    save_mode="postgres",
):
    print(chunk.text, end="", flush=True)
```

When `save_mode="postgres"` and `media_root` is configured, attached media files are automatically downloaded or copied into the structured folder `media_root/YYYY-MM/<response_id>/` and recorded in `responses.meta.media_files`.

## Architecture

```
XAIClient(mode="stream")
└─ StreamXAIClient.create_chat()
   ├─ xAIRequest (canonical domain model)
   ├─ xai_sdk.AsyncClient.chat.create(...)
   └─ generate_stream_and_persist() [chat_stream_xai.py]
      ├─ xai_stream() → official .stream()
      └─ persistence_manager.persist_request() + persist_response()
         (includes _save_media_files() when media is detected)
```

**Source files (current layout):**
- `src/ai_api/core/xai_client.py` – factory + `StreamXAIClient`
- `src/ai_api/core/xai/chat_stream_xai.py` – SDK orchestration + persistence integration
- `src/ai_api/core/xai/persistence_xai.py` – `xAIPersistenceManager` (infopypg + media handling)
- `src/ai_api/data_structures/xai_objects.py` – `LLMStreamingChunkProtocol`, `xAIRequest`, `xAIMessage`, etc.

## Persistence Guarantee

When `save_mode="postgres"`:
- A single request row is written **before** the streaming API call begins.
- A single response row is written **after** the stream completes (with accumulated text).
- Media files (if present) are saved to disk and their relative paths are stored in `responses.meta.media_files`.
- No per-chunk database writes – database load remains constant regardless of response length.

## Media File Handling (Unified)

Media files attached via `input_image` or `input_file` are automatically:
- Downloaded from HTTP(S) URLs.
- Copied from local absolute paths.
- Organised under `media_root/YYYY-MM/<response_id>/`.
- Logged in `media_root/index.txt` for auditing.
- Recorded in the PostgreSQL `responses.meta` column.

This behaviour is identical for both `mode="turn"` and `mode="stream"`.

## Logging

All streaming events, persistence actions, and media operations are logged at the appropriate level via the injected structured logger.

## Further Customisation

For additional details on configuration, structured output, batch processing, or error handling, refer to the top-level `README.md` or the relevant module docstrings.
