"""
json.py — File-based JSON persistence backend for the ai_api package.

This module implements the ``JsonFilePersistenceBackend`` class, which writes
timestamped neutral response blobs to the local filesystem.

**Design context and comparison with sibling modules**

See the module docstring of ``database.py`` for the overarching rationale.
In short, the three backends exist so that a client can be instantiated as the
immutable pair (model + persistence strategy) without runtime ``if`` switches
on ``save_mode``.

- ``database.py`` → durable, branch-aware, SQL analytics.
- ``json.py`` (this file) → simple, portable, zero-dependency archival.
- ``stdout.py`` → ephemeral, console-only output.

The JSON backend trades durability and branching for simplicity and portability.
It is ideal for offline reproducibility, air-gapped environments, or lightweight
logging where a full database would be overkill.

All three modules share an identical documentation template so that a developer
who understands one instantly understands the others.

**Error handling (updated 2026)**

Filesystem errors are wrapped via the single generic ``wrap_error(FilePersistenceError, ...)``.
This ensures every JSON write failure carries the precise custom type and
is logged exactly once.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg
import py_pgkit as pgk
from py_pgkit.db import PgSettings
from py_pgkit.db import get_pool as get_pgk_pool

from ..errors import wrap_error, FilePersistenceError

__all__ = ["JsonFilePersistenceBackend"]


class JsonFilePersistenceBackend:
    """Simple file-based backend that writes timestamped JSON blobs.

    Each call to ``persist`` creates a new file named
    ``response_<uuid>.json`` inside the configured directory.  The file
    contains the complete neutral blob plus metadata, making the entire
    interaction self-describing and portable.

    Parameters
    ----------
    json_dir : pathlib.Path or str, default "./persisted_requests"
        Directory where JSON files will be written.  Created automatically
        if it does not exist.
    logger : logging.Logger, optional
        Logger for diagnostics.  Defaults to a module-level logger.

    Attributes
    ----------
    json_dir : pathlib.Path
        Resolved directory used for all writes.
    logger : logging.Logger
        Active logger.

    See Also
    --------
    PostgresPersistenceBackend : durable, branch-aware alternative.
    StdoutPersistenceBackend : ephemeral console output.
    PersistenceManager.with_json_files : convenience constructor.

    Notes
    -----
    The backend is intentionally **stateless**.  No index or manifest is
    maintained; discovery of persisted interactions is performed by listing
    the directory or by external tooling.

    Examples
    --------
    >>> backend = JsonFilePersistenceBackend(json_dir="./my_logs")
    >>> meta = await backend.persist(response_blob, meta_dict)
    >>> meta["tree_id"] is None
    True
    # A new file has been written to ./my_logs/response_*.json
    """

    def __init__(
        self,
        json_dir: Path | str = "./persisted_requests",
        logger: logging.Logger | None = None,
    ) -> None:
        self.json_dir = Path(json_dir)
        self.logger = logger or logging.getLogger(__name__)

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
        """Write a timestamped JSON file containing the interaction.

        The file name is ``response_<uuid>.json`` so that concurrent writers
        never collide.  The payload is the union of the neutral blob and the
        supplied metadata, augmented with a ``created_at`` timestamp and the
        backend identifier.

        Any filesystem error is caught and converted via the single generic
        ``wrap_error(FilePersistenceError, ...)`` with full context and logging.

        Parameters
        ----------
        response_blob : dict
            The serialised ``NeutralResponseBlob``.
        meta : dict
            Metadata (model, usage, provider, …).  The key
            ``"backend": "json_files"`` is injected.
        tree_id, branch_id, ... : ignored
            Branching coordinates are accepted for API compatibility but are
            not stored (JSON backend has no notion of conversation graphs).
        kind : str, default "chat"
            Interaction type recorded in the file.

        Returns
        -------
        dict
            ``{"tree_id": None, "branch_id": None, "sequence": None,
            "parent_response_id": None, "kind": str}``
            All branching fields are ``None`` because the JSON backend does
            not participate in Git-style conversation trees.

        See Also
        --------
        database.PostgresPersistenceBackend.persist : the durable counterpart
            that *does* honour branching coordinates.
        """
        self.logger.debug(
            "Writing response to JSON file",
            extra={
                "model": meta.get("model"),
                "kind": kind,
                "json_dir": str(self.json_dir),
            },
        )
        try:
            self.json_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.json_dir / f"response_{uuid.uuid4()}.json"
            data = {
                "provider": meta.get("provider", "unknown"),
                "model": meta.get("model", "unknown"),
                "meta": {**meta, "backend": "json_files", "kind": kind},
                "payload": response_blob,
                "created_at": datetime.utcnow().isoformat(),
            }
            file_path.write_text(json.dumps(data, indent=2))
            self.logger.info(
                "Successfully wrote JSON persistence file",
                extra={
                    "path": str(file_path),
                    "model": meta.get("model"),
                    "kind": kind,
                },
            )
            return {
                "tree_id": None,
                "branch_id": None,
                "sequence": None,
                "parent_response_id": None,
                "kind": kind,
            }
        except Exception as exc:
            err = wrap_error(
                FilePersistenceError,
                "Failed to write JSON persistence file",
                exc,
                details={
                    "model": meta.get("model"),
                    "kind": kind,
                    "json_dir": str(self.json_dir),
                },
                logger=self.logger,
                level=logging.WARNING,
            )
            raise err from exc
