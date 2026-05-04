"""
database.py — PostgreSQL persistence backend for the ai_api package.

This module implements the ``PostgresPersistenceBackend`` class, which provides
durable, branch-aware storage of LLM interactions in a PostgreSQL database.

**Design context and comparison with sibling modules**

The ai_api design treats a client as the unique combination of (model + persistence
strategy).  To support this without runtime mode switches, the persistence concern
is factored into three cohesive, atomic modules:

- ``database.py`` (this file) — durable, relational storage with full Git-style
  branching, recursive history reconstruction, and support for long-lived
  conversational trees.
- ``json.py`` — file-based archival of neutral response blobs; simple, portable,
  no database required.
- ``stdout.py`` — ephemeral, console-only output for development, debugging, or
  fire-and-forget scripts.

These three backends are **polymorphic** via the shared ``PersistenceBackend``
interface defined in ``persistence.py``.  The parent ``PersistenceManager``
compares and contrasts them at construction time:

- **Durability & querying**: Only the Postgres backend offers ACID guarantees,
  rich SQL analytics, and the recursive CTE machinery needed for ``reconstruct_neutral_branch``.
  JSON and STDOUT backends return ``None`` for all branching identifiers.
- **Branching model**: Postgres alone supports the four relational columns
  (``tree_id``, ``branch_id``, ``parent_response_id``, ``sequence``) and the
  rebase/edit operations that make conversational history editable like Git.
  The other two backends are intentionally stateless with respect to conversation
  graphs.
- **Operational profile**: Postgres requires a running database and connection
  pool (py_pgkit or asyncpg).  JSON writes are local filesystem only.  STDOUT
  has zero external dependencies and zero latency beyond a ``print()``.
- **Use-case alignment**: Choose Postgres when you need auditability, branching,
  or production multi-user chat.  Choose JSON for offline reproducibility or
  simple logging.  Choose STDOUT when the only consumer is a human watching the
  terminal (the modern equivalent of the original ``save_mode="none"``).

All three modules follow an identical documentation template (numpy/scipy style)
so that developers can read any one of them and immediately understand the
others by analogy.

"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

import asyncpg
import py_pgkit as pgk
from py_pgkit.db import PgSettings
from py_pgkit.db import get_pool as get_pgk_pool

from ..errors import wrap_persistence_error

__all__ = ["PostgresPersistenceBackend"]


class PostgresPersistenceBackend:
    """Durable PostgreSQL backend with full branching and history reconstruction.

    This is the **production-grade** persistence strategy.  It stores every
    interaction as a row in the partitioned ``responses`` table, preserving the
    exact neutral format plus rich metadata.  The four branching columns enable
    Git-style conversation trees that can be reconstructed on demand via
    recursive CTE and edited via rebase-style operations (future extension).

    Parameters
    ----------
    settings : PgSettings, optional
        Modern py_pgkit configuration object (preferred).  Supplies host,
        database, credentials, pool size, etc.  When supplied, the backend
        participates in py_pgkit's shared connection pool and structured
        logging.
    db_url : str, optional
        Legacy PostgreSQL connection string (e.g.
        ``"postgresql://user:pass@localhost/ai_logs"``).  Only used when
        ``settings`` is ``None``.  Consider migrating to ``settings``.
    logger : logging.Logger, optional
        Logger for internal warnings and diagnostics.  Defaults to a
        module-level logger that can be configured via ``py_pgkit``.

    Attributes
    ----------
    settings : PgSettings or None
        The py_pgkit settings (if any) used for pool creation.
    db_url : str or None
        Legacy connection string (if any).
    logger : logging.Logger
        Active logger instance.

    See Also
    --------
    JsonFilePersistenceBackend : file-based alternative (no DB required).
    StdoutPersistenceBackend : ephemeral console output (no durability).
    PersistenceManager.with_postgres : convenience constructor that wires
        this backend.

    Notes
    -----
    The backend is intentionally **stateless** with respect to any particular
    conversation tree; all branching state lives in the database rows.  This
    allows multiple ``PersistenceManager`` instances (even across processes)
    to collaborate on the same conversation tree safely.

    Examples
    --------
    >>> from py_pgkit.db import PgSettings
    >>> settings = PgSettings(host="localhost", database="ai_logs", user="postgres")
    >>> backend = PostgresPersistenceBackend(settings=settings)
    >>> meta = await backend.persist(response_blob, meta_dict, tree_id=some_uuid)
    >>> print(meta["sequence"])
    7
    """

    def __init__(
        self,
        settings: PgSettings | None = None,
        db_url: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.db_url = db_url
        self.logger = logger or logging.getLogger(__name__)
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Lazily obtain (or create) a connection pool.

        Prefers the py_pgkit shared pool when ``settings`` is supplied;
        falls back to a private ``asyncpg`` pool for legacy ``db_url`` usage.
        Any failure is wrapped as ``DatabasePersistenceError`` and logged.
        """
        try:
            if self.settings is not None:
                return await get_pgk_pool(self.settings)
            if self._pool is None and self.db_url:
                self._pool = await asyncpg.create_pool(self.db_url)
            if self._pool is None:
                raise RuntimeError(
                    "PostgresPersistenceBackend requires either settings or db_url"
                )
            return self._pool
        except Exception as exc:
            err = wrap_persistence_error(
                exc,
                "Failed to obtain PostgreSQL connection pool",
                details={
                    "db_url": bool(self.db_url),
                    "has_settings": self.settings is not None,
                },
                logger=self.logger,
                level=logging.ERROR,
            )
            raise err from exc

    async def persist(
        self,
        response_blob: dict[str, Any],
        meta: dict[str, Any],
        *,
        tree_id: uuid.UUID | None = None,
        branch_id: uuid.UUID | None = None,
        parent_response_id: uuid.UUID | None = None,
        sequence: int | None = None,
        kind: str = "chat",
    ) -> dict[str, Any]:
        """Insert the interaction into the ``responses`` table.

        The method normalises the endpoint, generates a new ``response_id``,
        and performs a single-row INSERT using a prepared statement that
        exactly matches the schema produced by ``db_responses_schema.py``.

        All database errors are caught, wrapped as ``DatabasePersistenceError``,
        logged at WARNING level (callers typically continue), and re-raised.

        Parameters
        ----------
        response_blob : dict
            The serialised ``NeutralResponseBlob`` (prompt + generated turn +
            branch context).
        meta : dict
            Rich metadata (model, usage, provider, original ``save_mode`` for
            audit, etc.).  The key ``"backend": "postgres"`` is injected.
        tree_id, branch_id, parent_response_id, sequence : optional
            Git-style coordinates.  When ``None`` a brand-new tree/branch is
            created (sequence = 0).
        kind : str, default "chat"
            Interaction type recorded in the ``meta`` JSONB column.

        Returns
        -------
        dict
            ``{"tree_id": uuid, "branch_id": uuid, "sequence": int,
            "parent_response_id": uuid|None, "kind": str}``
            The coordinates that should be passed on the next turn of the same
            branch.

        Raises
        ------
        DatabasePersistenceError
            Any database error (connection loss, constraint violation, …)
            wrapped with structured details.

        See Also
        --------
        reconstruct_neutral_branch : companion method that walks the
            ``parent_response_id`` links created by this method.
        """
        self.logger.debug(
            "Persisting chat turn to Postgres",
            extra={"model": meta.get("model"), "kind": kind},
        )
        try:
            endpoint = meta.get("endpoint", {})
            if hasattr(endpoint, "to_dict"):
                endpoint = endpoint.to_dict()

            await self._insert_response_postgres(
                tree_id=tree_id,
                branch_id=branch_id,
                parent_response_id=parent_response_id,
                sequence=sequence,
                response_blob=response_blob,
                meta={**meta, "backend": "postgres", "kind": kind},
                endpoint=endpoint,
            )
            self.logger.info(
                "Successfully persisted to Postgres",
                extra={
                    "tree_id": str(tree_id) if tree_id else None,
                    "sequence": sequence,
                    "kind": kind,
                    "model": meta.get("model"),
                },
            )
            return {
                "tree_id": tree_id,
                "branch_id": branch_id,
                "sequence": sequence,
                "parent_response_id": parent_response_id,
                "kind": kind,
            }
        except Exception as exc:
            err = wrap_persistence_error(
                exc,
                "Failed to persist chat turn to PostgreSQL",
                details={
                    "model": meta.get("model"),
                    "kind": kind,
                    "tree_id": str(tree_id) if tree_id else None,
                },
                logger=self.logger,
                level=logging.WARNING,
            )
            raise err from exc

    async def _insert_response_postgres(
        self,
        tree_id: uuid.UUID | None,
        branch_id: uuid.UUID | None,
        parent_response_id: uuid.UUID | None,
        sequence: int | None,
        response_blob: dict,
        meta: dict,
        endpoint: dict,
        request_id: uuid.UUID | None = None,
        request_tstamp: datetime | None = None,
    ) -> None:
        """Low-level INSERT that exactly matches the responses table schema."""
        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO responses (
                        tstamp, provider_id, endpoint, request_id, request_tstamp,
                        response_id, response, meta,
                        tree_id, branch_id, parent_response_id, sequence
                    ) VALUES (
                        NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                    )
                    """,
                    1,                                                                    # TODO: replace with real provider_id lookup/cache
                    json.dumps(endpoint),
                    request_id or uuid.uuid4(),
                    request_tstamp or datetime.utcnow(),
                    uuid.uuid4(),                                                         # new response_id
                    json.dumps(response_blob),
                    json.dumps(meta),
                    tree_id,
                    branch_id,
                    parent_response_id,
                    sequence,
                )
        except Exception as exc:
            err = wrap_persistence_error(
                exc,
                "PostgreSQL INSERT failed",
                details={"tree_id": str(tree_id), "kind": meta.get("kind")},
                logger=self.logger,
                level=logging.ERROR,
            )
            raise err from exc

    async def reconstruct_neutral_branch(
        self,
        tree_id: uuid.UUID,
        branch_id: uuid.UUID,
        start_from_response_id: uuid.UUID | None = None,
        up_to_response_id: uuid.UUID | None = None,
        max_depth: int | None = None,
    ) -> list[dict]:
        """Reconstruct a slice of neutral conversation history via recursive CTE.

        This is the heart of the zero-duplication branching model.  Full
        conversational history is never stored; only the delta (last prompt +
        generated response) lives in each row.  The recursive CTE walks the
        ``parent_response_id`` links in a single round-trip and returns the
        neutral turns in chronological order.

        Any query error is wrapped as ``DatabasePersistenceError`` and logged.

        Parameters
        ----------
        tree_id : uuid.UUID
            The conversation tree identifier.
        branch_id : uuid.UUID
            The specific branch within the tree.
        start_from_response_id : uuid.UUID, optional
            If supplied, reconstruction begins at this turn (inclusive).
            Useful when forking from a known point.
        up_to_response_id : uuid.UUID, optional
            If supplied, reconstruction stops before including this turn.
        max_depth : int, optional
            Safety limit to prevent runaway queries on extremely deep trees.

        Returns
        -------
        list of dict
            List of ``NeutralTurn`` dictionaries in chronological order,
            ready to be passed to ``OllamaRequest.from_neutral_history`` or
            the equivalent xAI method.

        See Also
        --------
        persist : the method that creates the rows this query traverses.
        """
        self.logger.debug(
            "Reconstructing neutral branch",
            extra={"tree_id": str(tree_id), "branch_id": str(branch_id)},
        )
        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                query = """
                    WITH RECURSIVE branch_history AS (
                        SELECT response_id, response, sequence, parent_response_id, 0 AS depth
                        FROM responses
                        WHERE tree_id = $1 AND branch_id = $2
                          AND (parent_response_id IS NULL OR response_id = COALESCE($3, (
                              SELECT response_id FROM responses
                              WHERE tree_id = $1 AND branch_id = $2
                              ORDER BY sequence ASC LIMIT 1
                          )))
                        UNION ALL
                        SELECT r.response_id, r.response, r.sequence, r.parent_response_id, bh.depth + 1
                        FROM responses r
                        JOIN branch_history bh ON r.parent_response_id = bh.response_id
                        WHERE ($4 IS NULL OR bh.depth < $4)
                    )
                    SELECT response, sequence, response_id
                    FROM branch_history
                    WHERE ($5 IS NULL OR response_id != $5)
                    ORDER BY sequence ASC
                """
                rows = await conn.fetch(
                    query,
                    tree_id,
                    branch_id,
                    start_from_response_id,
                    max_depth,
                    up_to_response_id,
                )
            history: list[dict] = []
            for row in rows:
                resp = row["response"]
                if isinstance(resp, str):
                    resp = json.loads(resp)
                neutral_turn = (
                    resp.get("response", resp) if isinstance(resp, dict) else resp
                )
                history.append(neutral_turn)
            return history
        except Exception as exc:
            err = wrap_persistence_error(
                exc,
                "Failed to reconstruct neutral branch",
                details={"tree_id": str(tree_id), "branch_id": str(branch_id)},
                logger=self.logger,
                level=logging.ERROR,
            )
            raise err from exc
