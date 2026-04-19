**<FILE filename="README_TURNBASED.md" size="2284 bytes">**
# Turn-Based Functionality – xAIClient

## Overview

The `mode="turn"` path (default) in `XAIClient` provides a clean, stateful turn-based chat interface using the **official xAI Python SDK** (`xai_sdk.AsyncClient`).

**Key benefits:**
- Fully stateful conversation management via the official SDK chat object.
- Direct return of a fully typed `xAIResponse` (with `.text`, `.parsed`, `.tool_calls`, `.reasoning_text`, etc.).
- Complete support for all xAI features (reasoning traces, tools, prompt caching, structured JSON output).
- Unified multimodal handling identical to streaming mode.
- Single request/response persistence pattern for straightforward auditing.

Multimodal content (images and files) is now **fully unified** with streaming mode. There is no longer a separate `"multim"` mode.

## Quick Start

```python
from ai_api import XAIClient
from pathlib import Path

client = XAIClient(
    logger=logger,
    api_key="xai-...",
    mode="turn",                                      # default mode
    persistence_manager=xAIPersistenceManager(
        pg_resolved_settings=..., 
        media_root=Path("/path/to/media")             # ← enables automatic media saving
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

response = await client.create_chat(
    messages=messages,
    model="grok-4",
    temperature=0.7,
    save_mode="postgres",
    # response_model=YourPydanticModel   # optional structured output
)

print(response.text)
if response.parsed:
    print("Structured output:", response.parsed)
if response.reasoning_text:
    print("Reasoning trace:", response.reasoning_text)
```

When `save_mode="postgres"` and `media_root` is configured, attached media files are automatically downloaded or copied into the structured folder `media_root/YYYY-MM/<response_id>/` and recorded in `responses.meta.media_files`.

For multi-turn conversations, the returned `response` contains an internal `_sdk_chat` attribute that can be used by the SDK for seamless continuation (see module docstring for advanced usage).

## Architecture

```
XAIClient(mode="turn")
└─ TurnXAIClient.create_chat()
   ├─ xAIRequest (canonical domain model)
   ├─ xai_sdk.AsyncClient.chat.create(...)
   └─ create_turn_chat_session() [chat_turn_xai.py]
      ├─ persistence_manager.persist_request()
      ├─ SDK chat.sample()
      └─ persistence_manager.persist_response()
         (includes _save_media_files() when media is detected)
```

**Source files (current layout):**
- `src/ai_api/core/xai_client.py` – factory + `TurnXAIClient`
- `src/ai_api/core/xai/chat_turn_xai.py` – SDK orchestration + persistence integration
- `src/ai_api/core/xai/persistence_xai.py` – `xAIPersistenceManager` (infopypg + media handling)
- `src/ai_api/data_structures/xai_objects.py` – `xAIRequest`, `xAIResponse`, `xAIMessage`, `xAIJSONResponseSpec`, etc.

## Persistence Guarantee

When `save_mode="postgres"`:
- A single request row is written **before** the API call.
- A single response row is written **after** the call completes.
- Media files (if present) are saved to disk and their relative paths are stored in `responses.meta.media_files`.
- Structured output (when `response_model` is supplied) is automatically parsed and attached to `response.parsed`.

## Media File Handling (Unified)

Media files attached via `input_image` or `input_file` are automatically:
- Downloaded from HTTP(S) URLs.
- Copied from local absolute paths.
- Organised under `media_root/YYYY-MM/<response_id>/`.
- Logged in `media_root/index.txt` for auditing.
- Recorded in the PostgreSQL `responses.meta` column.

This behaviour is identical for both `mode="turn"` and `mode="stream"`.

## Logging

All turn-based events, persistence actions, and media operations are logged at the appropriate level via the injected structured logger.

## Further Customisation

For additional details on structured output, stateful multi-turn conversations, batch processing, streaming mode, or error handling, refer to the top-level `README.md` or the relevant module docstrings.
