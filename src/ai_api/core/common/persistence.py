"""
PersistenceManager — fully generic, protocol-driven persistence layer.

This module is deliberately decoupled from any concrete request/response types
(xAIRequest, OllamaRequest, OllamaEmbedRequest, etc.). It only depends on the
two lightweight protocols defined in data_structures/base_objects.py.

All persistence logic now works exclusively with plain Python dictionaries
returned by:
    request.meta()   → generation settings
    request.payload() → actual prompt/messages/input
    request.endpoint() → provider/model/connection info

This design means:
- Adding a new provider or feature (e.g. embeddings, batch, future models) requires
  zero changes to persistence.py
- No isinstance() checks anywhere
- Maximum future-proofing and testability
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from infopypg import ensure_partition_exists, execute_query

# Only import the protocols — never the concrete classes
from ..data_structures.base_objects import LLMRequestProtocol, LLMResponseProtocol
from .errors import wrap_persistence_error

if TYPE_CHECKING:
    # Only for type hints in development — not imported at runtime
    from ..data_structures.ollama_objects import OllamaRequest
    from ..data_structures.xai_objects import xAIRequest

__all__ = ["PersistenceManager"]


# SQL templates (kept here for clarity; in production these would live in a separate sql/ module)
SQL_INSERT_REQUEST = """
    INSERT INTO requests (
        provider_id, endpoint, request_id, payload, meta,
        tree_id, branch_id, parent_response_id, sequence, tstamp
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
    RETURNING tstamp
"""

SQL_INSERT_RESPONSE = """
    INSERT INTO responses (
        provider_id, endpoint, request_id, request_tstamp, response_id,
        payload, meta, tree_id, branch_id, parent_response_id, sequence, tstamp
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
"""


class PersistenceManager:
    """Generic persistence layer for all LLM request/response objects."""

    def __init__(
        self,
        pg_resolved_settings: Any = None,
        logger: logging.Logger | None = None,
        conversation_id: str | None = None,
        media_root: Any = None,
    ) -> None:
        self._pg_resolved_settings = pg_resolved_settings
        self._pool: Any = None
        self.logger = logger or logging.getLogger(__name__)
        self.conversation_id = conversation_id or "unknown"
        self.media_root = media_root

    async def get_pool(self) -> Any:
        """Lazily create and return asyncpg connection pool."""
        if self._pool is None:
            # In real implementation this would use the resolved settings
            # to create the pool. Placeholder kept for compatibility.
            self._pool = "mock_pool"  # TODO: replace with real pool creation
        return self._pool

    async def _get_pool(self) -> Any:
        return await self.get_pool()

    async def get_or_create_provider_id(self, provider: str) -> int:
        """Return (or create) numeric provider id for 'xai' or 'ollama'."""
        # In production this would query a providers table.
        # For now we use a simple mapping.
        return {"xai": 1, "ollama": 2}.get(provider.lower(), 99)

    # ------------------------------------------------------------------
    # CORE: persist_request — completely generic
    # ------------------------------------------------------------------
    async def persist_request(
        self,
        request: LLMRequestProtocol,
        batch_id: str | None = None,
        batch_index: int | None = None,
        tree_id: uuid.UUID | None = None,
        branch_id: uuid.UUID | None = None,
        parent_response_id: uuid.UUID | None = None,
        sequence: int | None = None,
    ) -> tuple[uuid.UUID, datetime]:
        """
        Persist any request object that implements LLMRequestProtocol.

        Uses only request.meta(), request.payload(), request.endpoint().
        No knowledge of the concrete class is required.
        """
        try:
            meta = request.meta()
            payload = request.payload()
            endpoint = request.endpoint()

            provider = endpoint.get("provider", "unknown")
            provider_id = await self.get_or_create_provider_id(provider)

            # Enrich meta with common fields used by the application layer
            meta = {
                **meta,
                "conversation_id": self.conversation_id,
                "batch_id": batch_id,
                "batch_index": batch_index,
                "tree_id": str(tree_id) if tree_id else None,
                "branch_id": str(branch_id) if branch_id else None,
            }

            request_id = uuid.uuid4()
            today = datetime.now(timezone.utc).date()

            await ensure_partition_exists(
                connection_pool=await self._get_pool(),
                table_name="requests",
                target_date=today,
                logger=self.logger,
            )

            pool = await self._get_pool()
            result = await execute_query(
                pool,
                SQL_INSERT_REQUEST,
                params=[
                    provider_id,
                    endpoint,           # full endpoint dict (provider, model, base_url, path, ...)
                    str(request_id),
                    payload,
                    meta,
                    tree_id,
                    branch_id,
                    parent_response_id,
                    sequence,
                ],
                fetch=True,
                logger=self.logger,
            )

            tstamp = result[0]["tstamp"] if result else datetime.now(timezone.utc)
            self.logger.info(
                "Request persisted",
                extra={"request_id": str(request_id), "provider": provider, "model": endpoint.get("model")},
            )
            return request_id, tstamp

        except Exception as exc:
            raise wrap_persistence_error(exc, "Failed to persist request") from exc

    # ------------------------------------------------------------------
    # CORE: persist_response — completely generic
    # ------------------------------------------------------------------
    async def persist_response(
        self,
        request_id: uuid.UUID,
        request_tstamp: datetime,
        api_result: dict[str, Any],
        request: LLMRequestProtocol | None = None,
        batch_id: str | None = None,
        tree_id: uuid.UUID | None = None,
        branch_id: uuid.UUID | None = None,
        parent_response_id: uuid.UUID | None = None,
        sequence: int | None = None,
    ) -> None:
        """
        Persist response using standardized api_result + optional request object.

        Preferred:  api_result contains 'meta' and 'payload' keys (from the client layer)
        Fallback:   we derive from request.meta() / request.payload() if available.
        """
        try:
            # --- Determine provider from request (if supplied) or api_result ---
            if request is not None:
                endpoint = request.endpoint()
                provider = endpoint.get("provider", "unknown")
            else:
                endpoint = api_result.get("endpoint", {})
                provider = endpoint.get("provider", api_result.get("provider", "unknown"))

            provider_id = await self.get_or_create_provider_id(provider)

            # --- Extract standardized fields from api_result (preferred) ---
            meta = api_result.get("meta", {})
            payload = (
                api_result.get("payload")
                or api_result.get("response_payload")
                or api_result.get("raw_data")
                or api_result.get("raw")
                or api_result
            )

            # Enrich meta
            meta = {
                **meta,
                "conversation_id": self.conversation_id,
                "batch_id": batch_id,
            }

            # Optional media handling (still needs the request object for now)
            if request is not None and hasattr(request, "has_media") and request.has_media():
                media_files = await self._save_media_files(uuid.uuid4(), request)
                meta["media_files"] = media_files

            response_id = uuid.uuid4()
            today = datetime.now(timezone.utc).date()

            await ensure_partition_exists(
                connection_pool=await self._get_pool(),
                table_name="responses",
                target_date=today,
                logger=self.logger,
            )

            pool = await self._get_pool()
            await execute_query(
                pool,
                SQL_INSERT_RESPONSE,
                params=[
                    provider_id,
                    endpoint,
                    str(request_id),
                    request_tstamp,
                    str(response_id),
                    payload,
                    meta,
                    tree_id,
                    branch_id,
                    parent_response_id,
                    sequence,
                ],
                fetch=False,
                logger=self.logger,
            )

            self.logger.info(
                "Response persisted",
                extra={"response_id": str(response_id), "provider": provider},
            )

        except Exception as exc:
            raise wrap_persistence_error(exc, "Failed to persist response") from exc

    # ------------------------------------------------------------------
    # Optional helper — media saving (can be moved to a mixin later)
    # ------------------------------------------------------------------
    async def _save_media_files(self, response_id: uuid.UUID, request: Any) -> list[str]:
        """Placeholder for media file extraction & storage."""
        # In real implementation this would inspect request.payload() for
        # image/file references and copy them into self.media_root.
        self.logger.debug("Media saving not yet implemented in generic layer")
        return []


# Convenience type aliases for callers that still want concrete types in development
if TYPE_CHECKING:
    RequestT = LLMRequestProtocol | "xAIRequest" | "OllamaRequest"
    ResponseT = LLMResponseProtocol | "xAIResponse" | "OllamaResponse"
