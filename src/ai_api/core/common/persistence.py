"""
PersistenceManager — single entry point for persisting every LLM interaction.

Modernized to use **py_pgkit** (replaces manual asyncpg pools + legacy "logger"/infopypg).

The PersistenceManager provides **one public method** — ``persist_chat_turn`` —
that is used for **all** interactions with LLMs: ordinary turn-based chats,
streaming chats (final response only), branched conversations, embeddings,
batch jobs, one-off completions, and any future interaction type.

Design goals (numpy/scipy style)
--------------------------------
- **Zero history duplication for chats**: only the delta (last prompt +
  generated response) is stored in the ``response`` JSONB column. Full
  conversational history is reconstructed on demand via the recursive CTE in
  ``reconstruct_neutral_branch``.
- **Git-style branching**: the four relational columns ``tree_id``,
  ``branch_id``, ``parent_response_id`` and ``sequence`` give every chat turn
  a precise location in an arbitrarily deep tree of conversations.
- **Provider-agnostic neutral format**: every provider-specific response
  object (``OllamaResponse``, ``xAIResponse``, …) implements
  ``to_neutral_format`` so the persistence layer never needs to know the
  concrete type.
- **Uniform treatment of non-chat use cases**: embeddings, batch items and
  one-off requests are persisted through the exact same code path by
  passing ``branching=False`` (or omitting tree/branch identifiers). They
  simply receive ``NULL`` in the branching columns and a ``kind`` value in
  ``meta``.
- **py_pgkit integration (NEW)**: Shared cached pools via ``get_pool(settings)``,
  optional high-performance ``bulk_insert`` for volume scenarios,
  structured logging via ``py_pgkit.logging`` (drop-in, optional DB backend
  to the ``logs`` table with ``obj`` JSONB), and convenient ``query_logs()``
  (replaces raw SQL on the partitioned logs table).

The schema defined in ``db_responses_schema.py`` (now using py_pgkit.DatabaseBuilder)
is sufficient; no additional tables or columns are required.

Error handling note
-------------------
All database and validation failures are wrapped using the helpers from
``.errors`` (``wrap_persistence_error``, ``wrap_database_error``, ``wrap_client_error``).
This ensures every exception that leaves this module is a rich ``AIAPIError``
subclass with ``__cause__`` and structured ``details``, while adding **zero**
overhead on the happy path.

See Also
--------
reconstruct_neutral_branch : companion method that rebuilds history via
    recursive CTE on ``parent_response_id``.
ChatSession : high-level convenience wrapper used by client code.
py_pgkit docs: shared pools, bulk_insert, query_logs, graceful shutdown.


Module level examples
---------------------
All examples assume an async context (or ``asyncio.run(...)``).

Turn-based chat (most common) — modern py_pgkit style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> from py_pgkit.db import PgSettings
>>> import py_pgkit as pgk
>>> settings = PgSettings(host="localhost", database="responsesdb", user="postgres", password=...)
>>> pgk.configure_logging(settings)  # optional: structured logs → Postgres
>>> logger = pgk.logging.getLogger(__name__)
>>> pm = PersistenceManager(logger=logger, settings=settings)  # or db_url=... for legacy
>>> req = OllamaRequest(model="llama3", input=..., temperature=0.7, save_mode="postgres")
>>> resp = await ollama_client._call_ollama(req)
>>> meta = await pm.persist_chat_turn(resp, req)
>>> print(meta["tree_id"], meta["sequence"])
uuid... 0

Streaming chat, branched, embeddings, batch — unchanged API, now benefits from
shared pools and py_pgkit logging.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import asyncpg
import py_pgkit as pgk
from py_pgkit.db import PgSettings, bulk_insert, query_logs
from py_pgkit.db import get_pool as get_pgk_pool

from ai_api.data_structures.base_objects import (
    LLMRequestProtocol,
    LLMResponseProtocol,
    NeutralResponseBlob,
    NeutralTurn,
)

# NEW: use the canonical error wrappers so this module participates in the
# rich, typed error system it was designed for.
from .errors import (
    wrap_client_error,
    wrap_database_error,
    wrap_persistence_error,
)

__all__: list[str] = ["PersistenceManager", "persist_batch_requests"]


class PersistenceManager:
    """Unified, branch-aware persistence layer for every LLM interaction type.

    Modernized (2026): now accepts ``settings: PgSettings`` (preferred) for
    py_pgkit shared pools, bulk operations, and structured DB logging.
    Backward compatible with ``db_url`` (uses direct asyncpg pool).

    This class replaces the older ``persist_request`` / ``persist_response``
    pair.  All call sites — chat, embeddings, batch, streaming, one-off —
    now go through the single public coroutine ``persist_chat_turn``.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        settings: PgSettings | None = None,
        db_url: str | None = None,
        media_root: Path | None = None,
        json_dir: Path | None = None,
    ) -> None:
        """Create a new persistence manager.

        Parameters
        ----------
        logger : logging.Logger | None
            Structured logger. If None, defaults to ``py_pgkit.logging.getLogger(__name__)``
            (drop-in replacement; call ``pgk.configure_logging(settings)`` first for DB backend).
        settings : PgSettings, optional
            Modern py_pgkit settings (Pydantic, env-var aware). Preferred.
            Enables shared cached pools, bulk_insert, query_logs, etc.
        db_url : str, optional
            Legacy PostgreSQL connection string (e.g. "postgresql://user:pass@host/db").
            Used only if ``settings`` is None. Consider migrating to ``settings``.
        media_root : pathlib.Path, optional
            Base directory for any media artifacts referenced by responses
            (images, audio, …).  Currently unused but reserved.
        json_dir : pathlib.Path, optional
            Directory for ``save_mode="json_files"`` output.
            Defaults to ``./persisted_requests``.
        """
        if logger is None:
            logger = pgk.logging.getLogger(__name__)
        self.logger = logger

        self.settings: PgSettings | None = settings
        self.db_url: str | None = db_url
        self.media_root = media_root
        self.json_dir = json_dir or Path("./persisted_requests")
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Lazily get (or create) a connection pool.

        Prefers py_pgkit's shared, cached pool (via ``get_pool(settings)``)
        when ``settings`` is provided. Falls back to direct asyncpg.create_pool
        only for legacy ``db_url`` usage. This enables pool sharing across
        the entire application (clients, other modules, etc.).
        """
        if self.settings is not None:
            return await get_pgk_pool(self.settings)

        if self._pool is None and self.db_url:
            self._pool = await asyncpg.create_pool(self.db_url)
        return self._pool                                                                 # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal JSON helpers (used when save_mode="json_files")
    # ------------------------------------------------------------------
    async def _persist_response_json(
        self, provider: str, model: str, meta: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        """Write a neutral response blob to a timestamped JSON file.

        The file name is ``response_<uuid>.json`` so that concurrent writers
        never collide.
        """
        self.json_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.json_dir / f"response_{uuid.uuid4()}.json"
        data = {
            "provider": provider,
            "model": model,
            "meta": meta,
            "payload": payload,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        file_path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # PUBLIC API — the single method everything now uses
    # ------------------------------------------------------------------
    async def persist_chat_turn(
        self,
        provider_response: LLMResponseProtocol,
        provider_request: LLMRequestProtocol | None = None,
        *,
        tree_id: uuid.UUID | None = None,
        branch_id: uuid.UUID | None = None,
        parent_response_id: uuid.UUID | None = None,
        sequence: int | None = None,
        neutral_history_slice: list[dict] | None = None,
        kind: Literal["chat", "embedding", "completion", "batch", "oneoff"] = "chat",
        branching: bool = True,
    ) -> dict[str, Any]:
        """Persist one LLM interaction (chat turn, embedding, batch item, …).

        This is the **only** public persistence method.  It:

        1. Calls the provider-specific ``to_neutral_format`` on the response
           object to obtain a ``NeutralTurn`` (text, structured output,
           tool calls, embedding vector, …).
        2. Builds a ``NeutralResponseBlob`` containing the exact prompt that
           produced the response (system + user) together with the generated
           turn.  Only the *delta* is stored — never the full history.
        3. Writes the blob to the ``response`` JSONB column and the
           branching metadata to the relational columns of the ``responses``
           table (or to a JSON file when ``save_mode="json_files"``).
        4. Returns the identifiers the caller needs to continue the same
           branch or to fork from this point.

        When ``branching=True`` (the default) and ``tree_id`` is ``None`` a
        brand-new tree/branch/sequence=0 is created.  When
        ``parent_response_id`` is supplied the new turn becomes its child.

        For embeddings, batch jobs or one-off completions set
        ``branching=False`` (or simply omit the tree/branch parameters).  The
        four branching columns receive ``NULL`` and a ``kind`` value is
        recorded in ``meta`` so that later queries can still distinguish
        interaction types.

        **py_pgkit benefits**: Uses shared pool (no per-instance pool creation),
        logger automatically participates in ``flush_all_handlers()`` if
        DB-backed logging is configured.

        Parameters
        ----------
        provider_response : LLMResponseProtocol
            Concrete response returned by the LLM (``OllamaResponse``,
            ``xAIResponse``, future embedding response objects, …).  Must
            implement ``to_neutral_format(branch_info)``.
        provider_request : LLMRequestProtocol, optional
            The request that produced ``provider_response``.  Used to
            extract the precise prompt (system message + last user turn)
            that will be stored inside the neutral blob.
        tree_id : uuid.UUID, optional
            Existing conversation tree.  Only meaningful when
            ``branching=True``.
        branch_id : uuid.UUID, optional
            Existing branch within the tree.  Only meaningful when
            ``branching=True``.
        parent_response_id : uuid.UUID, optional
            ``response_id`` of the turn this new turn should be attached to.
            Required for any continuation or fork.
        sequence : int, optional
            Explicit sequence number inside the branch.  If omitted it is
            computed as ``parent.sequence + 1`` (or 0 for a new branch).
        neutral_history_slice : list of dict, optional
            Only the relevant slice of already-reconstructed neutral history.
            Used solely to locate the last system message; the full history
            is never written to the database.
        kind : {"chat", "embedding", "completion", "batch", "oneoff"}, default "chat"
            Interaction type recorded in ``meta["kind"]``.  Used by
            analytics and to decide whether branching columns should be
            populated.
        branching : bool, default True
            If ``False`` the four branching columns are set to ``NULL``
            regardless of any identifiers passed by the caller.  Use for
            embeddings, batch items and one-off requests.

        Returns
        -------
        dict
            ``{"tree_id": uuid|None, "branch_id": uuid|None, "sequence": int|None,
            "parent_response_id": uuid|None, "kind": str}``
            The identifiers that should be passed on the next call if the
            caller wants to continue or fork this conversation (only
            populated when ``branching=True``).

        Raises
        ------
        AIPersistenceError
            Any failure during persistence (DB write, JSON handling, validation).
            The original exception is attached via ``__cause__``.
        """
        # ------------------------------------------------------------------
        # 1. Normalise branching for non-chat interactions
        # ------------------------------------------------------------------
        if not branching or kind != "chat":
            tree_id = None
            branch_id = None
            parent_response_id = None
            sequence = None

        elif tree_id is None:
            tree_id = uuid.uuid4()
            branch_id = uuid.uuid4()
            sequence = 0

        if sequence is None and branching:
            sequence = 1 if parent_response_id else 0

        # ------------------------------------------------------------------
        # 2. Build branch context (empty for non-branched interactions)
        # ------------------------------------------------------------------
        branch_info: dict[str, Any] = {}
        if branching and tree_id is not None:
            branch_info = {
                "tree_id": str(tree_id),
                "branch_id": str(branch_id),
                "parent_response_id": str(parent_response_id)
                if parent_response_id
                else None,
                "sequence": sequence,
            }

        # ------------------------------------------------------------------
        # 3. Convert provider response → neutral turn
        # ------------------------------------------------------------------
        try:
            neutral_resp_dict = provider_response.to_neutral_format(
                branch_info=branch_info
            )
        except Exception as exc:
            raise wrap_persistence_error(
                exc, "Failed to convert provider response to neutral format"
            ) from exc

        # ------------------------------------------------------------------
        # 4. Build the prompt that produced this response (delta only)
        # ------------------------------------------------------------------
        last_prompt = {
            "system": self._extract_last_system(neutral_history_slice or []),
            "user": self._extract_last_user_prompt(provider_request),
            "structured_spec": getattr(provider_request, "response_format", None)
            if provider_request
            else None,
            "tools": getattr(provider_request, "tools", None)
            if provider_request
            else None,
        }

        response_blob = NeutralResponseBlob(
            prompt=last_prompt,
            response=NeutralTurn(**neutral_resp_dict),
            branch_context=branch_info,
        ).model_dump(mode="json")

        # ------------------------------------------------------------------
        # 5. Assemble meta (everything that is not the neutral blob)
        # ------------------------------------------------------------------
        meta: dict[str, Any] = {
            **(provider_response.meta() if hasattr(provider_response, "meta") else {}),
            **(
                provider_request.meta()
                if provider_request and hasattr(provider_request, "meta")
                else {}
            ),
            "provider": getattr(provider_response.endpoint(), "provider", "unknown"),
            "model": getattr(provider_response.endpoint(), "model", "unknown"),
            "kind": kind,
            "branching": branching,
        }

        # ------------------------------------------------------------------
        # 6. Decide save mode and write
        # ------------------------------------------------------------------
        save_mode = getattr(
            provider_response,
            "save_mode",
            getattr(provider_request, "save_mode", "none")
            if provider_request
            else "none",
        )

        if save_mode == "postgres" and (self.settings or self.db_url):
            if hasattr(provider_response, "endpoint"):
                endpoint = getattr(provider_response, "endpoint", lambda: {})()
                if not isinstance(endpoint, dict):
                    endpoint = endpoint.to_dict()
            else:
                endpoint = {}

            try:
                await self._insert_response_postgres(
                    tree_id=tree_id,
                    branch_id=branch_id,
                    parent_response_id=parent_response_id,
                    sequence=sequence,
                    response_blob=response_blob,
                    meta=meta,
                    endpoint=endpoint,
                )
            except Exception as exc:
                raise wrap_database_error(
                    exc, "Failed to insert response into Postgres"
                ) from exc

        elif save_mode == "json_files":
            try:
                await self._persist_response_json(
                    provider_response.endpoint().provider
                    if hasattr(provider_response, "endpoint")
                    else "unknown",
                    meta.get("model", "unknown"),
                    meta,
                    response_blob,
                )
            except Exception as exc:
                raise wrap_persistence_error(
                    exc, "Failed to write response JSON file"
                ) from exc

        self.logger.info(
            "LLM interaction persisted",
            extra={
                "kind": kind,
                "tree_id": str(tree_id) if tree_id else None,
                "branch_id": str(branch_id) if branch_id else None,
                "sequence": sequence,
            },
        )

        return {
            "tree_id": tree_id,
            "branch_id": branch_id,
            "sequence": sequence,
            "parent_response_id": parent_response_id,
            "kind": kind,
        }

    # ------------------------------------------------------------------
    # RECONSTRUCTION
    # ------------------------------------------------------------------
    async def reconstruct_neutral_branch(
        self,
        tree_id: uuid.UUID,
        branch_id: uuid.UUID,
        start_from_response_id: uuid.UUID | None = None,
        up_to_response_id: uuid.UUID | None = None,
        max_depth: int | None = None,
    ) -> list[dict]:
        """Reconstruct a slice of neutral conversation history via recursive CTE.

        Walks the ``parent_response_id`` links in the ``responses`` table,
        starting either from the root of the branch or from a specific response
        (useful when forking).  Returns only the neutral turns (the
        ``response.response`` part of each ``NeutralResponseBlob``), already
        ordered chronologically.

        This method is the heart of the zero-duplication branching design:
        full history never needs to be stored in the JSONB blob; it is
        rebuilt on demand in a single database round-trip even for trees
        with thousands of turns.

        Uses the modern py_pgkit pool when available.

        Parameters
        ----------
        tree_id : uuid.UUID
            The conversation tree identifier.
        branch_id : uuid.UUID
            The specific branch within the tree.
        start_from_response_id : uuid.UUID, optional
            If provided, the walk begins at this response (i.e. the first
            returned turn will be the *child* of this response).  Used when
            forking from a particular point in an existing conversation.
        up_to_response_id : uuid.UUID, optional
            If provided, the walk stops before including the turn whose
            ``response_id`` equals this value.  Useful when you want history
            *up to but not including* a parent you are about to fork from.
        max_depth : int, optional
            Safety limit on the number of turns returned.  Prevents runaway
            queries on extremely deep or cyclic data (should never happen
            in normal use).

        Returns
        -------
        list of dict
            List of neutral turns (``NeutralTurn`` dicts) in chronological order.
            Each element is suitable for passing directly to
            ``OllamaRequest.from_neutral_history`` or the equivalent xAI method.

        Raises
        ------
        AIDatabaseError
            On any Postgres error during the recursive CTE query.
        """
        if not (self.settings or self.db_url):
            return []

        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                query = """
                    WITH RECURSIVE branch_history AS (
                        SELECT
                            response_id,
                            response,
                            sequence,
                            parent_response_id,
                            0 AS depth
                        FROM responses
                        WHERE tree_id = $1
                          AND branch_id = $2
                          AND (
                              parent_response_id IS NULL
                              OR response_id = COALESCE($3, (
                                  SELECT response_id FROM responses
                                  WHERE tree_id = $1 AND branch_id = $2
                                  ORDER BY sequence ASC LIMIT 1
                              ))
                          )

                        UNION ALL

                        SELECT
                            r.response_id,
                            r.response,
                            r.sequence,
                            r.parent_response_id,
                            bh.depth + 1
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
            raise wrap_database_error(
                exc, f"Failed to reconstruct branch {branch_id} (tree {tree_id})"
            ) from exc

    # ======================================================================
    # Git-style history editing (rebase) support
    # ======================================================================

    async def create_edited_branch(
        self,
        tree_id: uuid.UUID,
        source_branch_id: uuid.UUID,
        edit_ops: list[dict[str, Any]],
        *,
        new_branch_name: str | None = None,
        start_from_response_id: uuid.UUID | None = None,
        end_at_response_id: uuid.UUID | None = None,
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        """
        Create a new branch by applying edit operations to a slice of an existing branch.

        This is the primary high-level API for arbitrary history editing (remove turns,
        insert new turns, etc.). It follows the Git rebase model: the original branch
        is left completely unchanged; a new branch with new response_id values is created.

        Parameters
        ----------
        tree_id : uuid.UUID
            The conversation tree this branch belongs to.
        source_branch_id : uuid.UUID
            The branch to base the edit on.
        edit_ops : list of dict
            Ordered list of edit operations. Supported operations:
            - {"op": "remove_turns", "indices": [0, 2, 5]}
            - {"op": "insert_turn_after", "after_index": 3, "turn": NeutralTurn dict or minimal {"role", "content"}}
            - {"op": "replace_turn", "index": 4, "turn": {...}}
        new_branch_name : str, optional
            Human-readable name stored in Conversations.branch_metadata.
        start_from_response_id : uuid.UUID, optional
            If provided, reconstruction starts from this turn (inclusive).
        end_at_response_id : uuid.UUID, optional
            If provided, reconstruction stops before this turn.

        Returns
        -------
        dict
            {
                "new_branch_id": uuid.UUID,
                "new_response_ids": list[uuid.UUID],
                "edited_history": list[dict],
                "operations_applied": list[dict],
                "conversation_id": uuid.UUID | None,
            }

        Raises
        ------
        AIClientError
            If edit_ops would produce an empty history or invalid indices
            (client-side validation error).
        AIDatabaseError
            On any database error during the new branch creation.
        """
        if not self.db_url:
            raise wrap_client_error(
                RuntimeError("create_edited_branch requires a Postgres connection"),
                "Cannot create edited branch without database connection",
            )

        try:
            history: list[dict] = await self.reconstruct_neutral_branch(
                tree_id=tree_id,
                branch_id=source_branch_id,
                start_from_response_id=start_from_response_id,
                up_to_response_id=end_at_response_id,
                max_depth=max_depth or 1000,
            )
        except Exception as exc:
            raise wrap_persistence_error(
                exc, "Failed to load source branch for editing"
            ) from exc

        if not history:
            raise wrap_client_error(
                ValueError("Source branch slice is empty; nothing to edit"),
                "Cannot edit empty conversation history",
            )

        edited_history, applied_ops = self._apply_edit_operations(history, edit_ops)

        if not edited_history:
            raise wrap_client_error(
                ValueError("Edit operations resulted in empty history"),
                "Edit operations produced an empty conversation",
            )

        new_branch_id = uuid.uuid4()
        new_response_ids: list[uuid.UUID] = []
        new_parent_id: uuid.UUID | None = None

        for seq, neutral_turn in enumerate(edited_history):
            branch_info = {
                "tree_id": str(tree_id),
                "branch_id": str(new_branch_id),
                "parent_response_id": str(new_parent_id) if new_parent_id else None,
                "sequence": seq,
            }

            response_blob = {
                "prompt": {
                    "system": self._extract_last_system(edited_history[: seq + 1]),
                    "user": neutral_turn.get("content")
                    if neutral_turn.get("role") == "user"
                    else "",
                    "structured_spec": neutral_turn.get("structured"),
                    "tools": neutral_turn.get("tools"),
                },
                "response": neutral_turn,
                "branch_context": branch_info,
            }

            try:
                new_response_id = await self._persist_edited_turn(
                    tree_id=tree_id,
                    branch_id=new_branch_id,
                    parent_response_id=new_parent_id,
                    sequence=seq,
                    response_blob=response_blob,
                    meta={
                        "kind": "edited",
                        "edit_source_branch_id": str(source_branch_id),
                        "edit_operations": applied_ops if seq == 0 else None,
                        "original_turn_index": history.index(neutral_turn)
                        if neutral_turn in history
                        else None,
                    },
                    endpoint={},
                )
            except Exception as exc:
                raise wrap_database_error(
                    exc, f"Failed to persist edited turn {seq} of new branch"
                ) from exc

            new_response_ids.append(new_response_id)
            new_parent_id = new_response_id

        try:
            conversation_id = await self._ensure_conversation_and_set_active_branch(
                tree_id=tree_id,
                active_branch_id=new_branch_id,
                branch_name=new_branch_name or f"edited-{new_branch_id.hex[:8]}",
            )
        except Exception as exc:
            raise wrap_persistence_error(
                exc, "Failed to update conversation metadata after rebase"
            ) from exc

        self.logger.info(
            "Created edited branch via rebase",
            extra={
                "tree_id": str(tree_id),
                "source_branch_id": str(source_branch_id),
                "new_branch_id": str(new_branch_id),
                "turns_before": len(history),
                "turns_after": len(edited_history),
                "operations": len(applied_ops),
            },
        )

        return {
            "new_branch_id": new_branch_id,
            "new_response_ids": new_response_ids,
            "edited_history": edited_history,
            "operations_applied": applied_ops,
            "conversation_id": conversation_id,
        }

    def _apply_edit_operations(
        self, history: list[dict], edit_ops: list[dict[str, Any]]
    ) -> tuple[list[dict], list[dict]]:
        """
        Apply a sequence of edit operations to a neutral history list.

        This is a pure function (no DB side-effects). It returns the new history
        and a log of exactly which operations were applied (for audit + new branch meta).
        """
        result = history.copy()
        applied: list[dict] = []

        for op in edit_ops:
            op_type = op.get("op")

            if op_type == "remove_turns":
                indices: list[int] = sorted(op.get("indices", []), reverse=True)
                removed = []
                for idx in indices:
                    if 0 <= idx < len(result):
                        removed.append(result.pop(idx))
                applied.append(
                    {
                        "op": "remove_turns",
                        "indices": indices,
                        "removed_count": len(removed),
                    }
                )

            elif op_type == "insert_turn_after":
                after_idx: int = op.get("after_index", len(result) - 1)
                new_turn = op.get("turn", {})
                if isinstance(new_turn, dict) and "role" in new_turn:
                    if 0 <= after_idx < len(result):
                        result.insert(after_idx + 1, new_turn)
                    elif after_idx == -1:
                        result.insert(0, new_turn)
                    applied.append(
                        {
                            "op": "insert_turn_after",
                            "after_index": after_idx,
                            "role": new_turn.get("role"),
                        }
                    )

            elif op_type == "replace_turn":
                idx: int = op.get("index", -1)
                new_turn = op.get("turn", {})
                if 0 <= idx < len(result) and isinstance(new_turn, dict):
                    old = result[idx]
                    result[idx] = {**old, **new_turn}
                    applied.append(
                        {
                            "op": "replace_turn",
                            "index": idx,
                            "old_role": old.get("role"),
                        }
                    )

            else:
                self.logger.warning(f"Unknown edit operation: {op_type}")
                applied.append(
                    {"op": op_type, "status": "skipped", "reason": "unknown operation"}
                )

        return result, applied

    async def _persist_edited_turn(
        self,
        tree_id: uuid.UUID,
        branch_id: uuid.UUID,
        parent_response_id: uuid.UUID | None,
        sequence: int,
        response_blob: dict,
        meta: dict,
        endpoint: dict,
    ) -> uuid.UUID:
        """Persist one turn of an edited (rebased) branch.

        Creates a brand-new immutable row with a fresh ``response_id``.
        Called by ``create_edited_branch`` for every turn in the edited
        history. The original branch remains untouched.
        """
        pool = await self._get_pool()
        new_response_id = uuid.uuid4()

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
                    1,
                    json.dumps(endpoint),
                    uuid.uuid4(),
                    datetime.now(timezone.utc).isoformat(),
                    new_response_id,
                    json.dumps(response_blob),
                    json.dumps(meta),
                    tree_id,
                    branch_id,
                    parent_response_id,
                    sequence,
                )
            return new_response_id
        except Exception as exc:
            raise wrap_database_error(
                exc, f"Failed to insert edited turn {sequence}"
            ) from exc

    async def _ensure_conversation_and_set_active_branch(
        self,
        tree_id: uuid.UUID,
        active_branch_id: uuid.UUID,
        branch_name: str,
    ) -> uuid.UUID | None:
        """Ensure a ``Conversations`` row exists and point it at the active branch."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT conversation_id FROM conversations WHERE tree_id = $1 LIMIT 1",
                tree_id,
            )
            if row:
                conv_id = row["conversation_id"]
                await conn.execute(
                    """
                    UPDATE conversations
                    SET active_branch_id = $1,
                        updated_at = NOW(),
                        branch_metadata = COALESCE(branch_metadata, '{}'::jsonb) || $2::jsonb
                    WHERE conversation_id = $3
                    """,
                    active_branch_id,
                    json.dumps(
                        {
                            branch_name: {
                                "created_at": datetime.now(timezone.utc).isoformat()
                            }
                        }
                    ),
                    conv_id,
                )
                return conv_id
            else:
                new_conv_id = uuid.uuid4()
                await conn.execute(
                    """
                    INSERT INTO conversations (conversation_id, tree_id, active_branch_id, title, meta, branch_metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    new_conv_id,
                    tree_id,
                    active_branch_id,
                    f"Conversation {tree_id.hex[:8]}",
                    json.dumps({}),
                    json.dumps(
                        {
                            branch_name: {
                                "created_at": datetime.now(timezone.utc).isoformat()
                            }
                        }
                    ),
                )
                return new_conv_id

    # ------------------------------------------------------------------
    # High-volume bulk insert helper
    # ------------------------------------------------------------------
    async def bulk_persist_responses(self, records: list[dict[str, Any]]) -> None:
        """High-performance bulk insert of pre-built response rows via COPY."""
        if not self.settings:
            raise wrap_client_error(
                RuntimeError(
                    "bulk_persist_responses requires settings=PgSettings (no db_url fallback)"
                ),
                "Bulk persistence requires modern py_pgkit settings",
            )
        await bulk_insert("responses", records, self.settings)

    # ------------------------------------------------------------------
    # Log querying (Uses py_pgkit.query_logs instead of raw SQL)
    # ------------------------------------------------------------------
    async def query_logs(
        self,
        level: str | None = None,
        logger_name: str | None = None,
        start_time: str | datetime | None = None,
        end_time: str | datetime | None = None,
        limit: int = 100,
        order_by: str = "tstamp DESC",
        **extra_filters: Any,
    ) -> list[dict[str, Any]]:
        """Query the structured `logs` table using py_pgkit's convenient helper."""
        if not self.settings:
            raise wrap_client_error(
                RuntimeError(
                    "query_logs requires settings=PgSettings (no db_url fallback)"
                ),
                "Log querying requires modern py_pgkit settings",
            )

        def _to_iso(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val.isoformat()
            return str(val)

        return await query_logs(
            self.settings,
            level=level,
            logger_name=logger_name,
            start_time=_to_iso(start_time),
            end_time=_to_iso(end_time),
            limit=limit,
            order_by=order_by,
            **extra_filters,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
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
        """Insert a row into the ``responses`` table (partitioned, branched)."""
        pool = await self._get_pool()
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
                1,
                json.dumps(endpoint),
                request_id or uuid.uuid4(),
                request_tstamp or datetime.now(timezone.utc).isoformat(),
                uuid.uuid4(),
                json.dumps(response_blob),
                json.dumps(meta),
                tree_id,
                branch_id,
                parent_response_id,
                sequence,
            )

    def _extract_last_system(self, history: list[dict]) -> str | None:
        """Return the most recent system message from a neutral history slice."""
        for turn in reversed(history):
            if turn.get("role") == "system":
                return turn.get("content")
        return None

    def _extract_last_user_prompt(
        self, request: LLMRequestProtocol | None
    ) -> str | list[dict]:
        """Extract the last user prompt (or raw prompt) from a request object."""
        if not request:
            return ""
        payload = request.payload()
        messages = payload.get("messages", [])
        if messages:
            last = messages[-1]
            return last.get("content", "") if isinstance(last, dict) else str(last)
        return payload.get("prompt", "")


# ----------------------------------------------------------------------
# Batch helper (legacy shim — now also warns via py_pgkit logger if used)
# ----------------------------------------------------------------------
async def persist_batch_requests(
    manager: "PersistenceManager", requests: list[LLMRequestProtocol]
) -> list[tuple[uuid.UUID | None, datetime | None]]:
    """Persist a list of requests that will later be sent as a batch.

    In the new architecture pure request-only persistence is discouraged
    (the request is always persisted together with its response via
    ``persist_chat_turn``).  This helper therefore only logs a warning and
    returns placeholder identifiers so that existing call sites continue
    to work until they are migrated.
    """
    manager.logger.warning(
        "persist_batch_requests is a legacy shim; "
        "persist the responses with persist_chat_turn(kind='batch', branching=False) "
        "or use bulk_persist_responses for high volume"
    )
    return [(None, None) for _ in requests]
