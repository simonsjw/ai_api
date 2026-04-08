"""Media handler for Grok multimodal content.

This module defines the ``GrokMediaHandler`` class, which is responsible
for persisting images and files attached to Grok API requests and responses.

Media files are organised into a monthly/response_id folder structure under
the configured media root. An ``index.txt`` file is maintained for simple
auditing and reference purposes. All I/O operations (downloads and copies)
are executed asynchronously via ``asyncio.to_thread`` to avoid blocking the
event loop.
"""

from __future__ import annotations

import asyncio
import shutil
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError

# Local project imports
from ...data_structures.xai_objects import *

__all__: list[str] = [
    # Base exception
    "save_media_files",
]


async def save_media_files(
    self,
    response_id: uuid.UUID,
    request: xAIRequest,
) -> list[str]:
    """Save multimodal images/files to the monthly/response_id folder structure.
    Returns list of relative paths (from media root) for storage in responses.meta.
    """
    # Extract prompt snippet (first 100 chars of first user message)

    if not request.has_media():
        # Early exit for pure-text requests – no media to save
        self.logger.warning("No media found. Exiting helper.")
        return []

    prompt_snippet = request.extract_prompt_snippet(max_chars=100)

    media_items: list[dict[str, str]] = []
    for msg in request.input.messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if part.get("type") in ("input_image", "input_file"):
                    url_or_path = part.get("image_url") or part.get("file_url")
                    if url_or_path:
                        media_items.append(
                            {
                                "type": part["type"],
                                "url_or_path": url_or_path,
                                "original_name": Path(url_or_path).name or "file",
                            }
                        )

                        # Monthly folder
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    response_folder = self.media_root / month / str(response_id)
    response_folder.mkdir(parents=True, exist_ok=True)

    relative_paths: list[str] = []
    for item in media_items:
        src = item["url_or_path"]
        safe_name = Path(item["original_name"]).name
        dest_path = response_folder / safe_name

        try:
            if src.startswith(("http://", "https://")):
                # Download from URL
                def _download():
                    urllib.request.urlretrieve(src, str(dest_path))

                await asyncio.to_thread(_download)
            else:
                # Local file – copy
                await asyncio.to_thread(shutil.copy2, src, dest_path)
        except (URLError, OSError, FileNotFoundError) as exc:
            self.logger.warning(
                "Failed to save media file",
                extra={
                    "obj": {
                        "response_id": str(response_id),
                        "src": src,
                        "error": str(exc),
                    }
                },
            )
            continue

        relative_path = f"{month}/{response_id}/{safe_name}"
        relative_paths.append(relative_path)

        # Append to index.txt
        index_line = (
            f"{response_id}|"
            f"{datetime.now(timezone.utc).isoformat()}|"
            f"{relative_path}|"
            f"{prompt_snippet}\n"
        )
        index_path = self.media_root / "index.txt"

        def _append_index():
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with index_path.open("a", encoding="utf-8") as f:
                f.write(index_line)

        await asyncio.to_thread(_append_index)

    return relative_paths
