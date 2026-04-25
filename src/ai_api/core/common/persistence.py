"""
PersistenceManager — single entry point for persisting every LLM interaction.

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

The schema defined in ``db_responses_schema.py`` is sufficient; no
additional tables or columns are required.

Module level examples
---------------------
All examples assume an async context (or ``asyncio.run(...)``).

Turn-based chat (most common)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> pm = PersistenceManager(logger, db_url="postgresql://...")
>>> req = OllamaRequest(model="llama3", input=..., temperature=0.7, save_mode="postgres")
>>> resp = await ollama_client._call_ollama(req)          # returns OllamaResponse
>>> meta = await pm.persist_chat_turn(resp, req)          # new tree + branch created
>>> print(meta["tree_id"], meta["sequence"])
uuid... 0

Streaming chat (persist only the final accumulated response)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> # inside StreamOllamaClient.generate_stream_and_persist
>>> final_resp = OllamaResponse.from_dict(accumulated_chunks)
>>> meta = await pm.persist_chat_turn(
...     final_resp,
...     req,
...     tree_id=existing_tree,
...     branch_id=existing_branch,
...     parent_response_id=last_turn_id,
...     sequence=last_seq + 1,
... )

Branched / forked conversation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> # user clicked "fork from turn 3"
>>> history = await pm.reconstruct_neutral_branch(
...     tree_id, branch_id, start_from_response_id=turn3_id, max_depth=10
... )
>>> new_req = OllamaRequest.from_neutral_history(history, new_user_msg, meta_dict)
>>> new_resp = await client._call_ollama(new_req)
>>> meta = await pm.persist_chat_turn(
...     new_resp, new_req,
...     tree_id=tree_id, branch_id=new_uuid, parent_response_id=turn3_id
... )

Batch job (xAI or future provider)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> batch_reqs = [xAIRequest(...) for _ in range(100)]
>>> batch_resp = await xai_client.create_batch(batch_reqs)
>>> for i, resp in enumerate(batch_resp.responses):
...     await pm.persist_chat_turn(resp, batch_reqs[i], kind="batch", branching=False)

Embeddings (Ollama or xAI)
~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> emb_req = OllamaRequest(model="nomic-embed-text", input="hello world",
...                         kind="embedding")  # or via endpoint
>>> emb_resp = await ollama_client.embeddings(emb_req)
>>> await pm.persist_chat_turn(
...     emb_resp, emb_req, kind="embedding", branching=False
... )
# stored with kind="embedding" in meta, no tree/branch pollution

See Also
--------
reconstruct_neutral_branch : companion method that rebuilds history via
    recursive CTE on ``parent_response_id``.
ChatSession : high-level convenience wrapper used by client code.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import asyncpg

from ai_api.data_structures.base_objects import (
    LLMEndpoint,
    LLMRequestProtocol,
    LLMResponseProtocol,
    NeutralResponseBlob,
    NeutralTurn,
)

from ...data_structures.base_objects import SaveMode


class PersistenceManager:
    """Unified, branch-aware persistence layer for every LLM interaction type.

    This class replaces the older ``persist_request`` / ``persist_response``
    pair.  All call sites — chat, embeddings, batch, streaming, one-off —
    now go through the single public coroutine ``persist_chat_turn``.
    """

    def __init__(
        self,
        logger: logging.Logger,
        db_url: str | None = None,
        media_root: Path | None = None,
        json_dir: Path | None = None,
    ) -> None:
        """Create a new persistence manager.

        Parameters
        ----------
        logger : logging.Logger
            Structured logger (usually ``structlog`` or ``logging.getLogger``).
        db_url : str, optional
            PostgreSQL connection string.  If ``None`` only JSON-file
            persistence is available.
        media_root : pathlib.Path, optional
            Base directory for any media artifacts referenced by responses
            (images, audio, …).  Currently unused but reserved.
        json_dir : pathlib.Path, optional
            Directory for ``save_mode="json_files"`` output.
            Defaults to ``./persisted_requests``.
        """
        self.logger = logger
        self.db_url = db_url
        self.media_root = media_root
        self.json_dir = json_dir or Path("./persisted_requests")
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Lazily create (and cache) an asyncpg connection pool."""
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
            "payload": payload,                                                           # the NeutralResponseBlob
            "created_at": datetime.utcnow().isoformat(),
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
        AttributeError
            If ``provider_response`` does not implement ``to_neutral_format``.
        asyncpg.exceptions.PostgresError
            Any database error is propagated (connection loss, constraint
            violation, …).

        See Also
        --------
        reconstruct_neutral_branch : the method that later walks the
            ``parent_response_id`` links created by this method.
        ChatSession.create_or_continue : high-level wrapper used by the
            client factories.

        Notes
        -----
        The method is intentionally idempotent with respect to the neutral
        blob: calling it twice with the same objects produces two rows that
        differ only in ``tstamp`` and ``response_id``.  This is useful for
        retry logic after a transient DB failure.

        Examples
        --------
        Turn-based chat (new conversation)
        >>> req = OllamaRequest(
        ...     model="llama3", input=OllamaInput(...), temperature=0.7, save_mode="postgres"
        ... )
        >>> resp = await client._call_ollama(req)
        >>> meta = await pm.persist_chat_turn(resp, req)
        >>> meta["sequence"]
        0

        Streaming chat (persist final response only)
        >>> # inside the streaming loop you accumulate tokens
        >>> final = OllamaResponse.from_dict({"message": {"content": full_text}, ...})
        >>> meta = await pm.persist_chat_turn(
        ...     final,
        ...     req,
        ...     tree_id=ctx["tree_id"],
        ...     branch_id=ctx["branch_id"],
        ...     parent_response_id=ctx["last_id"],
        ...     sequence=ctx["seq"] + 1,
        ... )

        Forking a conversation
        >>> history = await pm.reconstruct_neutral_branch(
        ...     tree_id, branch_id, start_from_response_id=parent_id, max_depth=20
        ... )
        >>> new_req = OllamaRequest.from_neutral_history(history, new_prompt, meta)
        >>> new_resp = await client._call_ollama(new_req)
        >>> meta = await pm.persist_chat_turn(
        ...     new_resp,
        ...     new_req,
        ...     tree_id=tree_id,
        ...     branch_id=uuid.uuid4(),
        ...     parent_response_id=parent_id,
        ... )

        Batch embedding job
        >>> for text, emb_resp in zip(texts, embedding_responses):
        ...     await pm.persist_chat_turn(emb_resp, emb_req, kind="embedding", branching=False)
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
            # TODO: optionally insert row into Conversations table here

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
        neutral_resp_dict = provider_response.to_neutral_format(branch_info=branch_info)

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

        if save_mode == "postgres" and self.db_url:
            await self._insert_response_postgres(
                tree_id=tree_id,
                branch_id=branch_id,
                parent_response_id=parent_response_id,
                sequence=sequence,
                response_blob=response_blob,
                meta=meta,
                endpoint=getattr(provider_response, "endpoint", lambda: {})().to_dict()
                if hasattr(provider_response, "endpoint")
                else {},
            )
        elif save_mode == "json_files":
            await self._persist_response_json(
                provider_response.endpoint().provider
                if hasattr(provider_response, "endpoint")
                else "unknown",
                meta.get("model", "unknown"),
                meta,
                response_blob,
            )

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
    # RECONSTRUCTION (unchanged, already well documented)
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

        See Also
        --------
        persist_chat_turn : the method that writes new turns and populates
            the branching columns that this method later traverses.
        ChatSession.create_or_continue : high-level wrapper that calls both.
        """
        if not self.db_url:
            return []

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Recursive CTE that walks forward via parent_response_id links
            query = """
                WITH RECURSIVE branch_history AS (
                    -- Anchor: start from root of branch OR from a specific response
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

                    -- Recursive step: follow children (rows whose parent_response_id points to us)
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
                WHERE ($5 IS NULL OR response_id != $5)   -- exclude the up_to_response if provided
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
            # Extract the neutral turn (the part we actually want for history)
            neutral_turn = (
                resp.get("response", resp) if isinstance(resp, dict) else resp
            )
            history.append(neutral_turn)

        return history

    # ------------------------------------------------------------------
    # Low-level INSERT (matches db_responses_schema.py exactly)
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
        """Insert a row into the ``responses`` table (partitioned, branched).

        All parameters are passed positionally to the prepared statement so
        that the SQL stays identical to the one generated by
        ``db_responses_schema.py``.
        """
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
                1,                                                                        # TODO: resolve real provider_id from Providers table (lookup cache)
                json.dumps(endpoint),
                request_id or uuid.uuid4(),
                request_tstamp or datetime.utcnow(),
                uuid.uuid4(),                                                             # new response_id
                json.dumps(response_blob),
                json.dumps(meta),
                tree_id,
                branch_id,
                parent_response_id,
                sequence,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
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
# Batch helper
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

    Parameters
    ----------
    manager : PersistenceManager
        The persistence manager instance.
    requests : list of LLMRequestProtocol
        The batch of request objects.

    Returns
    -------
    list of (uuid.UUID | None, datetime | None)
        Placeholder identifiers (all ``(None, None)``).
    """
    manager.logger.warning(
        "persist_batch_requests is a legacy shim; "
        "persist the responses with persist_chat_turn(kind='batch', branching=False) instead"
    )
    return [(None, None) for _ in requests]
