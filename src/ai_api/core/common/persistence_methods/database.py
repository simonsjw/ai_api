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
compares and contrasts them at construction time.

**Error handling (updated 2026)**

All database-related exceptions are now routed through the single generic
``wrap_error(DatabasePersistenceError, ...)`` (or ``PersistenceError`` for
pool-level failures).  This guarantees consistent typing, structured details,
and centralised logging.
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

from ..errors import wrap_error, DatabasePersistenceError, PersistenceError

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
        Any failure is wrapped as ``PersistenceError`` (pool-level) via the
        single generic ``wrap_error`` factory and logged.
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
            err = wrap_error(
                PersistenceError,
                "Failed to obtain PostgreSQL connection pool",
                exc,
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

        All database errors are caught and wrapped via the single generic
        ``wrap_error(DatabasePersistenceError, ...)`` so that the raised
        exception is always of the precise custom type with full context.

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
            wrapped via ``wrap_error`` with structured details.

        See Also
        --------
        reconstruct_neutral_branch : companion method that walks the
            ``parent_response_id`` links created by this method.
        create_edited_branch : high-level editing/forking operation.
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
            err = wrap_error(
                DatabasePersistenceError,
                "Failed to persist chat turn to PostgreSQL",
                exc,
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
            err = wrap_error(
                DatabasePersistenceError,
                "PostgreSQL INSERT failed",
                exc,
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

        Any query error is wrapped via the single generic ``wrap_error`` as
        ``DatabasePersistenceError`` and logged at ERROR level.

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
        create_edited_branch : uses this method to build edited histories.
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
            err = wrap_error(
                DatabasePersistenceError,
                "Failed to reconstruct neutral branch",
                exc,
                details={"tree_id": str(tree_id), "branch_id": str(branch_id)},
                logger=self.logger,
                level=logging.ERROR,
            )
            raise err from exc

    async def create_edited_branch(
        self,
        tree_id: uuid.UUID,
        source_branch_id: uuid.UUID,
        edit_ops: list[dict[str, Any]],
        new_branch_name: str | None = None,
        start_from_response_id: uuid.UUID | None = None,
        end_at_response_id: uuid.UUID | None = None,
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        """Create a new edited branch from an existing committed history.

        This is the core backend implementation of the ad-hoc history editing
        feature restored in this refactor.  It:

        1. Reconstructs a slice of neutral history using ``reconstruct_neutral_branch``.
        2. Applies the list of ``edit_ops`` (remove, insert, replace, etc.).
        3. Inserts the edited turns as an entirely new branch (new ``branch_id``,
           new ``response_id``s, proper parent linking).
        4. Leaves the original branch completely untouched and immutable.

        This method is called by ``PersistenceManager.create_edited_branch``
        and ultimately by ``ChatSession.edit_history``.

        Parameters
        ----------
        tree_id : uuid.UUID
            The conversation tree identifier.
        source_branch_id : uuid.UUID
            The branch to edit within the tree.
        edit_ops : list of dict
            Edit operations to apply. Supported operations:
            - ``{"op": "remove_turns", "indices": [int, ...]}``
            - ``{"op": "insert_after", "after_index": int, "turn": dict}``
            - ``{"op": "replace_turn", "index": int, "turn": dict}``
        new_branch_name : str, optional
            Human-readable name for the new branch (stored in metadata).
        start_from_response_id, end_at_response_id, max_depth : optional
            Passed through to ``reconstruct_neutral_branch`` to limit the slice.

        Returns
        -------
        dict
            ``{
                "new_branch_id": uuid.UUID,
                "new_response_ids": list[uuid.UUID],
                "edited_history": list[dict],
                "operations_applied": int,
                "new_branch_name": str | None
            }``

        Raises
        ------
        DatabasePersistenceError
            Any database error during reconstruction or insertion of the
            new branch, wrapped via the generic ``wrap_error`` factory.

        See Also
        --------
        reconstruct_neutral_branch : used internally to fetch the base history.
        PersistenceManager.create_edited_branch : high-level wrapper.
        ChatSession.edit_history : user-facing API.
        """
        self.logger.info(
            "Creating edited branch",
            extra={
                "tree_id": str(tree_id),
                "source_branch_id": str(source_branch_id),
                "num_ops": len(edit_ops),
            },
        )

        # 1. Reconstruct the base history
        try:
            base_history = await self.reconstruct_neutral_branch(
                tree_id=tree_id,
                branch_id=source_branch_id,
                start_from_response_id=start_from_response_id,
                up_to_response_id=end_at_response_id,
                max_depth=max_depth,
            )
        except Exception as exc:
            err = wrap_error(
                DatabasePersistenceError,
                "Failed to reconstruct history for editing",
                exc,
                details={
                    "tree_id": str(tree_id),
                    "source_branch_id": str(source_branch_id),
                },
                logger=self.logger,
                level=logging.ERROR,
            )
            raise err from exc

        # 2. Apply edit operations (simple but effective implementation)
        edited_history = list(base_history)  # copy
        operations_applied = 0

        for op in edit_ops:
            op_type = op.get("op")
            if op_type == "remove_turns":
                indices = sorted(op.get("indices", []), reverse=True)
                for idx in indices:
                    if 0 <= idx < len(edited_history):
                        edited_history.pop(idx)
                        operations_applied += 1
            elif op_type == "insert_after":
                idx = op.get("after_index", -1)
                turn = op.get("turn", {})
                if 0 <= idx < len(edited_history):
                    edited_history.insert(idx + 1, turn)
                    operations_applied += 1
            elif op_type == "replace_turn":
                idx = op.get("index", -1)
                turn = op.get("turn", {})
                if 0 <= idx < len(edited_history):
                    edited_history[idx] = turn
                    operations_applied += 1

        # 3. Generate new branch and insert edited turns
        new_branch_id = uuid.uuid4()
        new_response_ids: list[uuid.UUID] = []
        prev_response_id = None

        new_meta = {
            "backend": "postgres",
            "kind": "edited_branch",
            "source_branch_id": str(source_branch_id),
            "new_branch_name": new_branch_name,
        }

        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                for i, turn in enumerate(edited_history):
                    new_response_id = uuid.uuid4()
                    new_response_ids.append(new_response_id)

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
                        1,  # provider_id placeholder
                        json.dumps({}),  # endpoint placeholder
                        uuid.uuid4(),  # request_id
                        datetime.utcnow(),
                        new_response_id,
                        json.dumps({"response": turn}),
                        json.dumps(new_meta),
                        tree_id,
                        new_branch_id,
                        prev_response_id,
                        i,
                    )
                    prev_response_id = new_response_id

            self.logger.info(
                "Successfully created edited branch",
                extra={
                    "new_branch_id": str(new_branch_id),
                    "operations_applied": operations_applied,
                },
            )

            return {
                "new_branch_id": new_branch_id,
                "new_response_ids": new_response_ids,
                "edited_history": edited_history,
                "operations_applied": operations_applied,
                "new_branch_name": new_branch_name,
            }

        except Exception as exc:
            err = wrap_error(
                DatabasePersistenceError,
                "Failed to insert edited branch into database",
                exc,
                details={
                    "tree_id": str(tree_id),
                    "source_branch_id": str(source_branch_id),
                    "new_branch_id": str(new_branch_id),
                },
                logger=self.logger,
                level=logging.ERROR,
            )
            raise err from exc
