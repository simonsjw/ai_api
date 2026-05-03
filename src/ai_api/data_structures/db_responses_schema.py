"""
Database schema for ai_api persistence – with Git-style conversation branching.

**What it does:**
Provides the complete SQLAlchemy declarative ORM layer for storing every
LLM request, response, log entry, and high-level conversation tree metadata
in a PostgreSQL database. The schema is heavily optimised for:

- Time-series partitioning (by ``tstamp``) so that even millions of
  interactions remain queryable.
- Git-like conversation branching: any response can become the parent of
  a new branch without duplicating content. This is achieved with
  ``tree_id``, ``branch_id``, ``parent_response_id`` and ``sequence``.
- Normalised providers table (only 5-10 rows) and rich JSONB columns for
  endpoint, request, response and meta so that schema evolution is trivial.

**How it does it:**
- All tables inherit from a standard SQLAlchemy ``declarative_base()``
  (py_pgkit handles table creation, extensions, partitioning, and tablespaces).
- Partitioning is declared with ``postgresql_partition_by = "RANGE (tstamp)"``.
- Composite primary keys + foreign-key constraints guarantee referential
  integrity even across partitions.
- The new ``Conversations`` table stores lightweight tree-level metadata
  (title, created/updated timestamps, arbitrary meta JSON) so that UI
  layers can list "conversations" without scanning the huge Requests table.
- Environment variable ``POSTGRES_DB_EXTENSIONS`` allows runtime addition of
  ``pg_trgm``, ``vector``, etc. without code changes.

The design deliberately keeps the original composite FK
(request_id + request_tstamp) unchanged for backward compatibility while
adding the branching fields as nullable columns.

Examples
--------
Typical usage (inside the package, not user code):

>>> from src.ai_api.data_structures.db_responses_schema import Requests, Responses
>>> # After a successful LLM call
>>> req_row = Requests(
...     provider_id=1,
...     endpoint={"provider": "ollama", "model": "llama3"},
...     request_id=uuid.uuid4(),
...     request={"messages": [...]},
...     meta={"temperature": 0.7},
...     tree_id=some_tree,
...     branch_id=some_branch,
...     sequence=0,
... )
>>> # Responses are linked via the composite FK and can point to a parent
>>> resp_row = Responses(
...     request_id=req_row.request_id,
...     request_tstamp=req_row.tstamp,
...     response_id=uuid.uuid4(),
...     response={"text": "Hello world"},
...     meta={"eval_count": 42},
...     parent_response_id=previous_resp_id,  # enables branching
...     sequence=1,
... )

See the ``main()`` function at the bottom for the one-time DB bootstrap
that creates the partitioned tables and required extensions.
"""

import asyncio
import uuid
from datetime import datetime
from os import getenv
from typing import Any

from py_pgkit.db import DatabaseBuilder, PgSettings
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

# Env override: Comma-separated, e.g., "pg_trgm,vector".
raw_ext: str | None = getenv("POSTGRES_DB_EXTENSIONS")
if raw_ext:
    extensions: list[str] = [ext.strip() for ext in raw_ext.split(",") if ext.strip()]
else:
    extensions = [
        "uuid-ossp",                                                                      # For potential UUIDs.
        "pg_trgm",                                                                        # For text search if needed.
    ]


class Providers(Base):
    """
    Lookup table for providers (normalization: 5-10 unique values).

    Parameters
    ----------
    id : int
        Auto-increment primary key.
    name : str
        Unique provider name (e.g. "ollama", "xai").
    description : str or None
        Human-readable description.

    Notes
    -----
    This table is populated once at bootstrap and then referenced by
    foreign key from Requests and Responses. Keeps the main tables small.
    """

    __tablename__: str = "providers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__: tuple[Index, dict[str, bool]] = (
        Index("ix_providers_name", "name"),
        {"extend_existing": True},
    )


Base = declarative_base()                                                                 # py_pgkit-compatible (no infopypg mixins needed)


class Requests(Base):
    """
    Main table for LLM requests (partitioned by tstamp).

    Stores the exact request payload, generation settings, and Git-style
    branching metadata. One row per logical LLM call.

    Parameters
    ----------
    idx : int
        Identity column (surrogate key for partitioning).
    tstamp : datetime
        Timestamp of the request (primary key component, used for
        partitioning).
    provider_id : int
        FK to Providers.
    endpoint : dict
        JSONB copy of the LLMEndpoint (provider, model, base_url, ...).
    request_id : uuid.UUID
        Stable identifier for the request (used in composite FK from Responses).
    request : dict
        The full request body that was sent to the model.
    meta : dict
        Generation parameters actually used (temperature, max_tokens, etc.).
    tree_id, branch_id, parent_response_id, sequence : uuid | int | None
        Git-style branching fields (nullable for legacy rows).

    See Also
    --------
    Responses : linked via composite FK (request_id, request_tstamp).
    """

    __tablename__: str = "requests"

    idx: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    tstamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), primary_key=True
    )
    provider_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("providers.id"), nullable=False
    )
    endpoint: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    request_id: Mapped[uuid.UUID] = mapped_column(pgUUID, nullable=False)
    request: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # ── Git-style branching fields (nullable for legacy data) ──
    tree_id: Mapped[uuid.UUID | None] = mapped_column(pgUUID, nullable=True, index=True)
    branch_id: Mapped[uuid.UUID | None] = mapped_column(
        pgUUID, nullable=True, index=True
    )
    parent_response_id: Mapped[uuid.UUID | None] = mapped_column(pgUUID, nullable=True)
    sequence: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__: tuple = (
        Index("ix_requests_tstamp", "tstamp"),
        Index("ix_requests_provider_id", "provider_id"),
        Index("ix_requests_request_id", "request_id"),
        Index("ix_requests_tree_id", "tree_id"),
        Index("ix_requests_branch_id", "branch_id"),
        UniqueConstraint("request_id", "tstamp", name="uq_requests_request_id_tstamp"),
        {"postgresql_partition_by": "RANGE (tstamp)", "extend_existing": True},
    )


class Responses(Base):
    """
    Main table for LLM responses (partitioned by tstamp).

    Mirrors Requests but adds the generated content, usage statistics,
    structured-output ``parsed`` field (if any), and branching pointers.

    Parameters
    ----------
    request_tstamp : datetime
        Timestamp of the originating request (part of composite FK).
    response_id : uuid.UUID
        Unique identifier for this response (used for self-referential FK
        when creating branches).
    response : dict
        The full raw response from the provider plus any parsed/structured
        output.
    meta : dict
        Usage + telemetry (tokens, duration, finish_reason, etc.).
    parent_response_id : uuid or None
        Points to the response that this branch diverged from.
    sequence : int or None
        Monotonic counter within a branch for deterministic ordering.

    Notes
    -----
    The self-referential FK uses ``ON DELETE SET NULL`` so that deleting a
    parent response does not cascade-delete its children (they become
    orphaned roots of new trees if needed).
    """

    __tablename__: str = "responses"

    idx: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    tstamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), primary_key=True
    )
    provider_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("providers.id"), nullable=False
    )
    endpoint: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    request_id: Mapped[uuid.UUID] = mapped_column(pgUUID, nullable=False)
    request_tstamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    response_id: Mapped[uuid.UUID] = mapped_column(pgUUID, nullable=False)
    response: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # ── Git-style branching fields (nullable for legacy data) ──
    tree_id: Mapped[uuid.UUID | None] = mapped_column(pgUUID, nullable=True, index=True)
    branch_id: Mapped[uuid.UUID | None] = mapped_column(
        pgUUID, nullable=True, index=True
    )
    parent_response_id: Mapped[uuid.UUID | None] = mapped_column(pgUUID, nullable=True)
    sequence: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__: tuple = (
        # === ORIGINAL INDEXES (kept unchanged) ===
        Index("ix_responses_tstamp", "tstamp"),
        Index("ix_responses_provider_id", "provider_id"),
        Index("ix_responses_request_id", "request_id"),
        Index("ix_responses_response_id", "response_id"),
        Index("ix_responses_request_id_tstamp", "request_id", "request_tstamp"),
        # === NEW BRANCHING INDEXES ===
        Index("ix_responses_tree_id", "tree_id"),
        Index("ix_responses_branch_id", "branch_id"),
        Index("ix_responses_parent_response_id", "parent_response_id"),
        # === ORIGINAL COMPOSITE FK (kept completely unchanged) ===
        ForeignKeyConstraint(
            columns=["request_id", "request_tstamp"],
            refcolumns=["requests.request_id", "requests.tstamp"],
            name="fk_responses_request_id_tstamp",
            ondelete="CASCADE",                                                           # CASCADE deletes response if request is deleted
        ),
        # === NEW SELF-REFERENTIAL FK FOR BRANCHING ===
        ForeignKeyConstraint(
            columns=["parent_response_id"],
            refcolumns=["responses.response_id"],
            name="fk_responses_parent_response_id",
            ondelete="SET NULL",                                                          # Safe for branching
        ),
        # === NEW UNIQUE CONSTRAINT FOR CLEAN BRANCH ORDERING ===
        UniqueConstraint("branch_id", "sequence", name="uq_responses_branch_sequence"),
        # === PARTITIONING (kept unchanged) ===
        {"postgresql_partition_by": "RANGE (tstamp)", "extend_existing": True},
    )


class Logs(Base):
    """
    Table for logging records (partitioned by tstamp).

    Captures structured log events (level, logger name, message, optional
    JSON object) for debugging, auditing, and post-mortem analysis of
    LLM interactions.

    Parameters
    ----------
    loglvl : str
        Log level ("DEBUG", "INFO", "WARNING", "ERROR").
    logger : str
        Name of the logger that emitted the record.
    message : str
        Human-readable log message.
    obj : dict or None
        Arbitrary JSON payload (e.g. request meta, exception traceback).
    """

    __tablename__: str = "logs"

    idx: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    tstamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), primary_key=True
    )
    loglvl: Mapped[str] = mapped_column(Text, nullable=False)
    logger: Mapped[str] = mapped_column(Text, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    obj: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__: tuple[Index, dict[str, str | bool]] = (
        Index("ix_logs_tstamp", "tstamp"),
        {"postgresql_partition_by": "RANGE (tstamp)", "extend_existing": True},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Metadata (lightweight tree-level information)
# ─────────────────────────────────────────────────────────────────────────────
class Conversations(Base):
    """
    High-level metadata for each chat *tree*.

    One row per logical conversation (supports multiple branches). This table
    is intentionally small and is the primary table queried by UI/dashboard
    code when listing a user's conversations.

    Parameters
    ----------
    conversation_id : uuid.UUID
        Primary key (also acts as stable external identifier).
    tree_id : uuid.UUID
        The root identifier shared by all branches of this conversation.
    title : str or None
        Optional human-readable title (can be auto-generated or user-edited).
    created_at, updated_at : datetime
        Timestamps for sorting and cache invalidation.
    meta : dict
        Arbitrary JSON (tags, user_id, project, etc.).
    """

    __tablename__: str = "conversations"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID, primary_key=True, default=uuid.uuid4
    )
    tree_id: Mapped[uuid.UUID] = mapped_column(pgUUID, nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now()
    )
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__: tuple = (
        Index("ix_conversations_tree_id", "tree_id"),
        Index("ix_conversations_conversation_id", "conversation_id"),
        {"extend_existing": True},
    )


async def main() -> None:
    """Bootstrap the database using py_pgkit (creates partitioned tables,
    extensions, tablespace, and registers all models).

    Uses PgSettings (env-var + Pydantic) and DatabaseBuilder.
    Call once at deploy time (or via `python -m ...`).

    Environment variables used
    --------------------------
    POSTGRES_U_POSTGRES_PW
        Password for the postgres superuser.
    POSTGRES_DB_EXTENSIONS
        Optional comma-separated list of extensions to create.

    After this, use:
        import py_pgkit as pgk
        pgk.configure_logging(settings)  # enables structured DB logging to `logs` table

    """
    settings = PgSettings(
        host="127.0.0.1",
        port=5432,
        user="postgres",
        password=getenv("POSTGRES_U_POSTGRES_PW"),
        database="responsesdb",
        extensions=extensions,
        # tablespace_name / tablespace_path supported via builder or env if extended
    )
    builder = DatabaseBuilder(
        settings=settings,
        models=[Base],                                                                    # or [Providers, Requests, Responses, Logs, Conversations]
        # partition_strategy="daily" optional if you want py_pgkit auto-partition helpers
    )
    await builder.build()

    print("Database initialised with Git-style conversation branching support.")


if __name__ == "__main__":
    asyncio.run(main())
