"""
Unified, protocol-based persistence layer for all LLM providers.

This module implements `PersistenceManager`, a single class that can persist
both requests and responses for any provider (Ollama, xAI, future providers)
without knowing their concrete types.

What it does
------------
- Saves `LLMRequestProtocol` and `LLMResponseProtocol` objects to either
  PostgreSQL (via asyncpg) or JSON files on disk, controlled by the
  `save_mode` attribute on the protocol objects.
- Provides a batch helper `persist_batch_requests` for providers that
  support batching (currently used by xAI batch mode).
- Fully decouples the persistence layer from any provider-specific objects
  by relying exclusively on the three methods defined in the protocols
  (`meta()`, `payload()`, `endpoint()`).

How it does it
--------------
- On construction you pass a logger, optional `db_url`, `media_root`, and
  `json_dir`.
- `persist_request` and `persist_response` inspect the `save_mode`
  (Literal["none", "json_files", "postgres"]) carried by the protocol object.
- When `save_mode == "postgres"` it uses an asyncpg connection pool (lazily
  created) and inserts into `llm_requests` / `llm_responses` tables.
- When `save_mode == "json_files"` it writes pretty-printed JSON files
  under `json_dir`.
- The design is deliberately symmetrical: both request and response sides
  use the same protocol methods. The optional `request` argument to
  `persist_response` exists only for future linking / correlation (currently
  just logged).

Examples — usage by clients
---------------------------
Typical usage inside a client (e.g. `StreamOllamaClient` or `xai_client.py`):

.. code-block:: python

    from ai_api.core.common.persistence import PersistenceManager
    from ai_api.data_structures.base_objects import SaveMode
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    pm = PersistenceManager(
        logger=logger,
        db_url="postgresql://user:pass@localhost:5432/ai_logs",
        json_dir=Path("./persisted_llm"),
    )

    # Inside a chat implementation
    request = OllamaRequest(...)  # implements LLMRequestProtocol
    rid, ts = await pm.persist_request(request)

    response = OllamaResponse(...)  # implements LLMResponseProtocol
    await pm.persist_response(response, request=request)

    # Batch example (xAI)
    from ai_api.core.common.persistence import persist_batch_requests

    results = await persist_batch_requests(pm, list_of_requests)

See Also
--------
ai_api.data_structures.base_objects.LLMRequestProtocol
ai_api.data_structures.base_objects.LLMResponseProtocol
    The minimal contracts that make this module provider-agnostic.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

from ai_api.data_structures.base_objects import (
    LLMEndpoint,
    LLMRequestProtocol,
    LLMResponseProtocol,
)

from ...data_structures.base_objects import SaveMode


class PersistenceManager:
    """Unified, fully symmetrical persistence layer for all LLM types.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance used for info/warning messages.
    db_url : str or None, optional
        PostgreSQL connection string. If None, Postgres persistence is disabled.
    media_root : pathlib.Path or None, optional
        Reserved for future media/blob storage (images, audio, etc.).
    json_dir : pathlib.Path or None, optional
        Directory for JSON file persistence. Defaults to ``./persisted_requests``.

    Attributes
    ----------
    logger : logging.Logger
    db_url : str or None
    media_root : pathlib.Path or None
    json_dir : pathlib.Path
    _pool : asyncpg.Pool or None
        Lazily created connection pool.
    """

    def __init__(
        self,
        logger: logging.Logger,
        db_url: str | None = None,
        media_root: Path | None = None,
        json_dir: Path | None = None,
    ) -> None:
        self.logger = logger
        self.db_url = db_url
        self.media_root = media_root
        self.json_dir = json_dir or Path("./persisted_requests")
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Lazily create and return the asyncpg connection pool."""
        if self._pool is None and self.db_url:
            self._pool = await asyncpg.create_pool(self.db_url)
        return self._pool                                                                 # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API — fully symmetrical
    # ------------------------------------------------------------------
    async def persist_request(
        self, request: LLMRequestProtocol
    ) -> tuple[uuid.UUID | None, datetime | None]:
        """Persist any request using the protocol (chat, embeddings, future providers).

        The request object must implement `LLMRequestProtocol` (i.e. provide
        `meta()`, `payload()`, and `endpoint()`). The `save_mode` is read from
        the request (falls back to "none").

        Parameters
        ----------
        request : LLMRequestProtocol
            Any object satisfying the request protocol.

        Returns
        -------
        tuple[uuid.UUID or None, datetime or None]
            The generated request_id and timestamp if persisted, otherwise (None, None).
        """
        save_mode = getattr(request, "save_mode", "none")
        if save_mode == "none":
            return None, None

        endpoint = request.endpoint()
        if isinstance(endpoint, LLMEndpoint):
            provider = endpoint.provider
            model = endpoint.model
            base_url = endpoint.base_url
        else:
            provider = endpoint.get("provider", "unknown")
            model = endpoint.get("model", "unknown")
            base_url = endpoint.get("base_url")

        meta = request.meta()
        payload = request.payload()

        request_id = uuid.uuid4()
        tstamp = datetime.utcnow()

        if save_mode == "postgres" and self.db_url:
            await self._persist_request_postgres(
                request_id, tstamp, provider, model, meta, payload, base_url
            )
        elif save_mode == "json_files":
            await self._persist_request_json(
                request_id, tstamp, provider, model, meta, payload
            )

        self.logger.info(
            "Request persisted",
            extra={"request_id": str(request_id), "provider": provider, "model": model},
        )
        return request_id, tstamp

    async def persist_response(
        self,
        response: LLMResponseProtocol,
        request: LLMRequestProtocol | None = None,
    ) -> None:
        """
        Persist any response using the protocol.

        This is now fully symmetrical with `persist_request`.
        The optional `request` parameter provides context (e.g. to link response
        to original request_id or to inherit save_mode if the response object
        itself doesn't carry it). Currently the link is only logged; full
        correlation is planned for a future schema update.

        Parameters
        ----------
        response : LLMResponseProtocol
            Any object satisfying the response protocol.
        request : LLMRequestProtocol or None, optional
            The original request (used only for context / future linking).
        """
        # Determine save_mode (prefer response if it has it, else fall back to request)
        save_mode = getattr(response, "save_mode", None)
        if save_mode is None and request is not None:
            save_mode = getattr(request, "save_mode", "none")

        if save_mode == "none":
            return

        endpoint = response.endpoint()
        if isinstance(endpoint, LLMEndpoint):
            provider = endpoint.provider
            model = endpoint.model
        else:
            provider = endpoint.get("provider", "unknown")
            model = endpoint.get("model", "unknown")

            # The response object itself provides meta + payload (fully protocol-driven)
        meta = response.meta()
        payload = response.payload()

        # If we have the original request, we can link them
        request_id = None
        if request is not None:
            # In a real implementation you would store the request_id from persist_request
            # For now we just log it
            pass

        if save_mode == "postgres" and self.db_url:
            await self._persist_response_postgres(provider, model, meta, payload)
        elif save_mode == "json_files":
            await self._persist_response_json(provider, model, meta, payload)

        self.logger.info(
            "Response persisted", extra={"provider": provider, "model": model}
        )

        # ------------------------------------------------------------------
        # Internal implementations
        # ------------------------------------------------------------------

    async def _persist_request_postgres(
        self,
        request_id: uuid.UUID,
        tstamp: datetime,
        provider: str,
        model: str,
        meta: dict[str, Any],
        payload: dict[str, Any],
        base_url: str | None,
    ) -> None:
        """Insert a single request row into the llm_requests table."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_requests (id, created_at, provider, model, meta, payload, base_url)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                request_id,
                tstamp,
                provider,
                model,
                json.dumps(meta),
                json.dumps(payload),
                base_url,
            )

    async def _persist_request_json(
        self,
        request_id: uuid.UUID,
        tstamp: datetime,
        provider: str,
        model: str,
        meta: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        """Write a single request as a pretty-printed JSON file."""
        self.json_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.json_dir / f"{request_id}.json"
        data = {
            "id": str(request_id),
            "created_at": tstamp.isoformat(),
            "provider": provider,
            "model": model,
            "meta": meta,
            "payload": payload,
        }
        file_path.write_text(json.dumps(data, indent=2))

    async def _persist_response_postgres(
        self,
        provider: str,
        model: str,
        meta: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        """Insert a single response row into the llm_responses table."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_responses (provider, model, meta, payload, created_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                provider,
                model,
                json.dumps(meta),
                json.dumps(payload),
                datetime.utcnow(),
            )

    async def _persist_response_json(
        self,
        provider: str,
        model: str,
        meta: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        """Write a single response as a pretty-printed JSON file."""
        self.json_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.json_dir / f"response_{uuid.uuid4()}.json"
        data = {
            "provider": provider,
            "model": model,
            "meta": meta,
            "payload": payload,
            "created_at": datetime.utcnow().isoformat(),
        }
        file_path.write_text(json.dumps(data, indent=2))

        # ------------------------------------------------------------------
        # Batch persistence helper (used by chat_batch_xai.py)
        # ------------------------------------------------------------------


async def persist_batch_requests(
    manager: "PersistenceManager",
    requests: list["LLMRequestProtocol"],
) -> list[tuple[uuid.UUID | None, datetime | None]]:
    """Persist a list of requests (works for xAI batch or future batch providers).

    This is a thin convenience wrapper around repeated calls to
    `manager.persist_request`. It is primarily used by the xAI batch client.

    Parameters
    ----------
    manager : PersistenceManager
        An initialised persistence manager.
    requests : list[LLMRequestProtocol]
        List of request objects (any that satisfy the protocol).

    Returns
    -------
    list[tuple[uuid.UUID or None, datetime or None]]
        Results from each individual persist_request call.
    """
    results = []
    for req in requests:
        rid, ts = await manager.persist_request(req)
        results.append((rid, ts))
    return results
