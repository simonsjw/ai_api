"""
persistence.py — Single point of entry for all LLM interaction persistence.

This module defines the abstract ``PersistenceBackend`` interface and the
``PersistenceManager`` orchestrator.  It is the *only* module that client code
should import directly; the three concrete implementations live in sibling
modules:

- ``database.py`` → ``PostgresPersistenceBackend``
- ``json.py``     → ``JsonFilePersistenceBackend``
- ``stdout.py``   → ``StdoutPersistenceBackend``

**Error handling model (updated 2026)**

All low-level exceptions raised by any backend are caught and converted
using the single generic factory ``wrap_error`` from ``.errors``.  This
ensures every error is an instance of the most appropriate custom type
(``PersistenceError``, ``DatabasePersistenceError``, etc.), with structured
``details`` and the original exception chained via ``__cause__``.

This file also exposes the high-level ``create_edited_branch`` operation
that was restored in the previous refactor commit.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal, Protocol

import py_pgkit as pgk
from py_pgkit.db import PgSettings

from ai_api.data_structures.base_objects import (
    LLMRequestProtocol,
    LLMResponseProtocol,
    NeutralResponseBlob,
    NeutralTurn,
)

from .errors import wrap_error, PersistenceError
from .persistence_methods.database import PostgresPersistenceBackend
from .persistence_methods.json import JsonFilePersistenceBackend
from .persistence_methods.stdout import StdoutPersistenceBackend

__all__ = [
    "PersistenceBackend",
    "PersistenceManager",
]


# =============================================================================
# Abstract Interface (defined here so all backends are measured against it)
# =============================================================================


class PersistenceBackend(Protocol):
    """Structural interface (Protocol) that all concrete backends satisfy.

    Using ``Protocol`` (structural subtyping) means the concrete backend classes
    do **not** need to inherit from this class — they are automatically
    recognised as ``PersistenceBackend`` as long as they implement the
    ``persist`` method with a compatible signature.

    This avoids circular imports while still giving the type checker (Pyrefly)
    full awareness that ``PostgresPersistenceBackend``, ``JsonFilePersistenceBackend``,
    and ``StdoutPersistenceBackend`` are all valid ``PersistenceBackend`` instances.

    Concrete implementations live in the sibling modules listed below.
    """

    async def persist(
        self,
        response_blob: dict[str, Any],
        meta: dict[str, Any],
        *,
        tree_id: uuid.UUID | None = None,
        branch_id: uuid.UUID | None = None,
        parent_response_id: uuid.UUID | None = None,
        sequence: int | None = None,
        kind: Literal["chat", "embedding", "completion", "batch", "oneoff"] = "chat",
    ) -> dict[str, Any]:
        """Persist (or echo) the interaction and return routing metadata.

        Parameters
        ----------
        response_blob : dict
            Serialised ``NeutralResponseBlob`` (prompt + generated turn +
            branch_context).
        meta : dict
            Full metadata (usage, model, provider, original ``save_mode`` if
            present for audit, …).  Backends may augment this.
        tree_id, branch_id, parent_response_id, sequence : optional
            Git-style branching coordinates.  Backends that do not support
            branching return ``None`` for these fields.
        kind : {"chat", "embedding", ...}, default "chat"
            Interaction type recorded for analytics.

        Returns
        -------
        dict
            ``{"tree_id": uuid|None, "branch_id": uuid|None, "sequence": int|None,
            "parent_response_id": uuid|None, "kind": str}``
        """
        ...


# =============================================================================
# PersistenceManager — the single orchestrator
# =============================================================================


class PersistenceManager:
    """Orchestrator that builds neutral blobs and delegates to a chosen backend.

    The manager is deliberately thin.  Its only responsibilities are:

    1. Normalise branching coordinates.
    2. Convert the provider response into a ``NeutralTurn`` via
       ``to_neutral_format``.
    3. Assemble a minimal ``NeutralResponseBlob`` (the *delta* only).
    4. Delegate the actual write/echo to the backend supplied at construction.

    Because the backend is injected at construction time, the higher-level
    client can be instantiated as the immutable pair (model + persistence
    strategy) with zero runtime branching on ``save_mode``.

    Recommended construction (compare the three options):

    >>> pm = PersistenceManager.with_postgres(settings=...)  # durable + branching
    >>> pm = PersistenceManager.with_json_files(json_dir="./logs")  # portable files
    >>> pm = PersistenceManager.with_stdout()  # console only (no side-effects)

    All three calls return a ``PersistenceManager`` that exposes the identical
    public API, satisfying the "one class per (model, persistence)" principle.

    Parameters
    ----------
    backend : PersistenceBackend
        The concrete strategy (Postgres, JSON, or STDOUT).
    logger : logging.Logger, optional
        Structured logger.  Defaults to a py_pgkit-aware logger.
    media_root : pathlib.Path, optional
        Reserved for future media artefact storage.

    See Also
    --------
    database.PostgresPersistenceBackend : durable implementation.
    json.JsonFilePersistenceBackend : file implementation.
    stdout.StdoutPersistenceBackend : console implementation.
    """

    def __init__(
        self,
        backend: PersistenceBackend,
        logger: logging.Logger | None = None,
        media_root: Any | None = None,
    ) -> None:
        self.backend = backend
        self.logger = logger or pgk.logging.getLogger(__name__)
        self.media_root = media_root

    # ------------------------------------------------------------------
    # Convenience constructors — the public surface for client wiring
    # ------------------------------------------------------------------

    @classmethod
    def with_postgres(
        cls,
        settings: PgSettings | None = None,
        db_url: str | None = None,
        logger: logging.Logger | None = None,
        media_root: Any | None = None,
    ) -> "PersistenceManager":
        """Return a manager that writes to PostgreSQL (durable + branching)."""
        backend = PostgresPersistenceBackend(
            settings=settings, db_url=db_url, logger=logger
        )
        return cls(backend=backend, logger=logger, media_root=media_root)

    @classmethod
    def with_json_files(
        cls,
        json_dir: Any = "./persisted_requests",
        logger: logging.Logger | None = None,
        media_root: Any | None = None,
    ) -> "PersistenceManager":
        """Return a manager that writes timestamped JSON files (portable)."""
        backend = JsonFilePersistenceBackend(json_dir=json_dir, logger=logger)
        return cls(backend=backend, logger=logger, media_root=media_root)

    @classmethod
    def with_stdout(
        cls,
        logger: logging.Logger | None = None,
        media_root: Any | None = None,
    ) -> "PersistenceManager":
        """Return a manager that echoes every response to STDOUT (ephemeral)."""
        backend = StdoutPersistenceBackend(logger=logger)
        return cls(backend=backend, logger=logger, media_root=media_root)

    # ------------------------------------------------------------------
    # Public API — the only method client code should call
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
        """Persist (or echo) one LLM interaction and return routing metadata.

        This is the **single public entry point** used by every client method
        (``create_chat``, ``ChatSession.create_or_continue``, …).

        The method performs the provider-agnostic work (neutral blob assembly)
        then delegates the I/O decision to the backend chosen at construction.
        Consequently, the caller never needs to know whether the output went to
        Postgres, a JSON file, or the console.

        Any exception raised by the backend is wrapped via the single generic
        ``wrap_error(PersistenceError, ...)`` so that every error carries the
        correct custom type, structured details, and is logged exactly once.

        Parameters
        ----------
        provider_response : LLMResponseProtocol
            Concrete response returned by the LLM (must implement
            ``to_neutral_format``).
        provider_request : LLMRequestProtocol, optional
            The originating request (used to extract the exact prompt for the
            blob and to preserve any ``save_mode`` value for audit metadata).
        tree_id, branch_id, parent_response_id, sequence : optional
            Branching coordinates.  Only honoured by the Postgres backend.
        neutral_history_slice : list of dict, optional
            Already-reconstructed slice used solely to locate the last system
            message.
        kind : {"chat", "embedding", ...}, default "chat"
            Interaction type stored in meta.
        branching : bool, default True
            Set ``False`` for embeddings / batch items.

        Returns
        -------
        dict
            Routing identifiers returned by the chosen backend.  For JSON and
            STDOUT backends all branching fields are ``None``.

        Raises
        ------
        PersistenceError
            Any error from the backend, wrapped with ``message`` and ``details``
            via the generic ``wrap_error`` factory.

        See Also
        --------
        database.PostgresPersistenceBackend.persist
        json.JsonFilePersistenceBackend.persist
        stdout.StdoutPersistenceBackend.persist

        Notes
        -----
        The original ``save_mode`` value (if present on the request or
        response) is written into ``meta["save_mode"]`` purely for auditability.
        It has no effect on routing — that decision was made when the
        ``PersistenceManager`` was constructed.

        Examples
        --------
        Using the STDOUT backend (the modern "none" semantics)
        >>> pm = PersistenceManager.with_stdout()
        >>> resp = await client.create_chat(messages=[{"role": "user", "content": "Hi"}])
        >>> meta = await pm.persist_chat_turn(resp, request)
        >>> meta["tree_id"] is None
        True
        # The generated text has already been printed to the console.
        """
        # 1. Normalise branching for non-chat interactions
        if not branching or kind not in ("chat",):
            tree_id = branch_id = parent_response_id = sequence = None
        elif tree_id is None:
            tree_id = uuid.uuid4()
            branch_id = uuid.uuid4()
            sequence = 0
        if sequence is None and branching:
            sequence = 1 if parent_response_id else 0

        # 2. Build branch context (only meaningful for Postgres)
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

        # 3. Provider response → neutral turn
        neutral_resp_dict = provider_response.to_neutral_format(branch_info=branch_info)

        # 4. Build prompt delta
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

        # 5. Assemble meta (preserve original save_mode only for audit)
        save_mode = getattr(
            provider_response,
            "save_mode",
            getattr(provider_request, "save_mode", None) if provider_request else None,
        )
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
            "save_mode": save_mode,                                                       # kept only for audit / legacy compatibility
            "backend": type(self.backend).__name__,
        }

        # 6. Delegate — wrap any backend error using the generic factory
        try:
            result = await self.backend.persist(
                response_blob=response_blob,
                meta=meta,
                tree_id=tree_id,
                branch_id=branch_id,
                parent_response_id=parent_response_id,
                sequence=sequence,
                kind=kind,
            )
            self.logger.info(
                "LLM interaction processed via %s backend",
                type(self.backend).__name__,
                extra={"kind": kind, "tree_id": str(tree_id) if tree_id else None},
            )
            return result
        except Exception as exc:
            err = wrap_error(
                PersistenceError,
                "Persistence backend failed",
                exc,
                details={
                    "backend": type(self.backend).__name__,
                    "model": meta.get("model"),
                    "kind": kind,
                },
                logger=self.logger,
                level=logging.WARNING,
            )
            raise err from exc

    # ------------------------------------------------------------------
    # High-level branching / editing operations (restored functionality)
    # ------------------------------------------------------------------

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

        This is the high-level operation that restores the ad-hoc history
        editing capability lost in the refactor.  It reconstructs a slice
        of neutral history, applies the supplied edit operations, and
        inserts the result as an entirely new branch (leaving the original
        branch untouched and immutable).

        This is the method called by ``ChatSession.edit_history``.

        Parameters
        ----------
        tree_id : uuid.UUID
            The conversation tree to operate on.
        source_branch_id : uuid.UUID
            The branch within the tree to edit.
        edit_ops : list of dict
            List of edit operations. Supported ops include:
            - ``{"op": "remove_turns", "indices": [int, ...]}``
            - ``{"op": "insert_after", "after_index": int, "turn": dict}``
            - ``{"op": "replace_turn", "index": int, "turn": dict}``
        new_branch_name : str, optional
            Human-readable name stored in branch metadata.
        start_from_response_id, end_at_response_id, max_depth : optional
            Slice parameters passed through to ``reconstruct_neutral_branch``.

        Returns
        -------
        dict
            ``{"new_branch_id": uuid, "new_response_ids": list[uuid],
            "edited_history": list[dict], "operations_applied": int, ...}``

        Raises
        ------
        DatabasePersistenceError
            Any database error during reconstruction or insertion,
            wrapped via the generic ``wrap_error`` factory.

        See Also
        --------
        database.PostgresPersistenceBackend.create_edited_branch
        ChatSession.edit_history
        """
        if not isinstance(self.backend, PostgresPersistenceBackend):
            raise PersistenceError(
                "create_edited_branch is only supported on the Postgres backend",
                details={"backend": type(self.backend).__name__},
            )

        try:
            result = await self.backend.create_edited_branch(
                tree_id=tree_id,
                source_branch_id=source_branch_id,
                edit_ops=edit_ops,
                new_branch_name=new_branch_name,
                start_from_response_id=start_from_response_id,
                end_at_response_id=end_at_response_id,
                max_depth=max_depth,
            )
            self.logger.info(
                "Created edited branch",
                extra={
                    "tree_id": str(tree_id),
                    "source_branch_id": str(source_branch_id),
                    "new_branch_id": str(result.get("new_branch_id")),
                },
            )
            return result
        except Exception as exc:
            err = wrap_error(
                PersistenceError,
                "Failed to create edited branch",
                exc,
                details={
                    "tree_id": str(tree_id),
                    "source_branch_id": str(source_branch_id),
                },
                logger=self.logger,
                level=logging.ERROR,
            )
            raise err from exc

    # ------------------------------------------------------------------
    # Private helpers (shared by all backends)
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
