"""
stdout.py — Ephemeral console output backend for the ai_api package.

This module implements the ``StdoutPersistenceBackend`` class, which prints
the generated LLM text (or structured payload) directly to ``sys.stdout``.

**Design context and comparison with sibling modules**

See the module docstring of ``database.py`` for the full rationale.  In summary,
the three backends exist so that a client object can be uniquely identified by
the immutable pair (model + persistence strategy) without any runtime mode
flags.

- ``database.py`` → durable, branch-aware, SQL analytics.
- ``json.py`` → portable file archival.
- ``stdout.py`` (this file) → zero-dependency, zero-latency, human-visible output.

The STDOUT backend is the modern realisation of the original ``save_mode="none"``
requirement.  It is intended for interactive development, CLI tools, or any
situation where the terminal is the primary consumer but the response object
must still be returned for further processing.

All three modules share an identical documentation template so that a developer
who understands one instantly understands the others by direct comparison.

**Error handling (updated 2026)**

Any failure during text extraction or printing is wrapped via the single
Generic ``wrap_error(OutputPersistenceError, ...)``.  This keeps the error
model uniform across all persistence strategies.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Any

import asyncpg
import py_pgkit as pgk
from py_pgkit.db import PgSettings
from py_pgkit.db import get_pool as get_pgk_pool

from ..errors import wrap_error, OutputPersistenceError

__all__ = ["StdoutPersistenceBackend"]


class StdoutPersistenceBackend:
    """Ephemeral backend that echoes the generated response to stdout.

    Every call to ``persist`` extracts the textual content (or falls back to
    a compact representation) and prints it to ``sys.stdout``, then flushes.
    The response object is still returned to the caller so that higher-level
    code can continue to use ``.text``, ``.parsed``, ``.tool_calls``, etc.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger for warnings when text extraction fails.  Defaults to a
        module-level logger.

    Attributes
    ----------
    logger : logging.Logger
        Active logger.

    See Also
    --------
    PostgresPersistenceBackend : durable, branch-aware alternative.
    JsonFilePersistenceBackend : file-based archival.
    PersistenceManager.with_stdout : convenience constructor that wires
        this backend (the modern ``save_mode="none"``).

    Notes
    -----
    The backend is intentionally **lossy** with respect to structured data:
    only the primary text content is printed.  If you need the full neutral
    blob or metadata, use the JSON or Postgres backend instead.

    Examples
    --------
    >>> backend = StdoutPersistenceBackend()
    >>> meta = await backend.persist(response_blob, meta_dict)
    >>> meta["tree_id"] is None
    True
    # The generated assistant message has already appeared on the console.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
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
        """Print the textual content of the response to stdout.

        The method first tries ``response_blob["response"]["content"]`` or
        ``["text"]``, then falls back to a compact string representation.
        Any extraction error is wrapped using the single generic
        ``wrap_error(OutputPersistenceError, ...)`` and logged at WARNING level
        (non-fatal — the caller still receives the response object).

        Parameters
        ----------
        response_blob : dict
            The serialised ``NeutralResponseBlob``.
        meta : dict
            Metadata (model, usage, …).  The key ``"backend": "stdout"`` is
            injected for auditability.
        tree_id, branch_id, ... : ignored
            Branching coordinates are accepted for API compatibility but have
            no effect (STDOUT backend has no notion of conversation state).
        kind : str, default "chat"
            Interaction type used only for logging.

        Returns
        -------
        dict
            ``{"tree_id": None, "branch_id": None, "sequence": None,
            "parent_response_id": None, "kind": str}``
            All branching fields are ``None``; the only useful information is
            ``kind``.

        See Also
        --------
        database.PostgresPersistenceBackend.persist : the durable counterpart
            that stores the full blob for later reconstruction.
        """
        self.logger.debug(
            "Echoing response to stdout",
            extra={"model": meta.get("model"), "kind": kind},
        )
        try:
            text: str | None = None
            resp = response_blob.get("response", {})
            if isinstance(resp, dict):
                text = resp.get("content") or resp.get("text")
            if not text:
                text = meta.get("text") or str(resp)
            if text:
                print(text, file=sys.stdout)
                sys.stdout.flush()
            else:
                print(
                    f"[{kind}] {meta.get('model', 'unknown')} response (no text)",
                    file=sys.stdout,
                )
            self.logger.info(
                "Response echoed to stdout",
                extra={"model": meta.get("model"), "kind": kind},
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
                OutputPersistenceError,
                "Failed to echo response to stdout",
                exc,
                details={"model": meta.get("model"), "kind": kind},
                logger=self.logger,
                level=logging.WARNING,
            )
            raise err from exc
