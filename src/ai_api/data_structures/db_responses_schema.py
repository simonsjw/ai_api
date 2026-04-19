#!/usr/bin/env python3
"""
Database schema for ai_api persistence – with Git-style conversation branching (refined April 2026).

Key refinements:
- Added lightweight Conversations table for tree-level metadata.
- Extended Requests and Responses with tree_id, branch_id, parent_response_id, sequence.
- Removed redundant conversation_id (tree_id is sufficient and cleaner).
- Preserved original composite FK (request_id + request_tstamp) unchanged.
- Added proper self-referential FK, unique constraint on (branch_id, sequence), and indexes.
- All new fields are nullable for safe migration of existing data.

This design enables arbitrary branching (like Git) with zero content duplication.
"""

import asyncio
import uuid
from datetime import datetime
from os import getenv
from typing import Any

from infopypg import Base, DatabaseBuilder
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
from sqlalchemy.orm import Mapped, mapped_column

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
    """

    __tablename__: str = "providers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__: tuple[Index, dict[str, bool]] = (
        Index("ix_providers_name", "name"),
        {"extend_existing": True},
    )


class Requests(Base):
    """
    Main table for LLM requests (partitioned by tstamp).
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
    # NEW: Conversation Metadata (lightweight tree-level information)
    # ─────────────────────────────────────────────────────────────────────────────


class Conversations(Base):
    """
    High-level metadata for each chat *tree*.
    One row per logical conversation (supports multiple branches).
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
    # Build (connects to 'postgres' for creation).
    settings = {
        "db_user": "postgres",
        "db_host": "127.0.0.1",
        "db_port": "5432",
        "db_name": "responsesdb",
        "password": getenv("POSTGRES_U_POSTGRES_PW"),
        "tablespace_name": "responses_db",
        "tablespace_path": "/mnt/HDD03_HIT_03TB/no_backup/pg03/responses_db",
        "extensions": extensions,
    }
    builder: DatabaseBuilder = DatabaseBuilder(settings)
    await builder.build()

    print("Database initialised with Git-style conversation branching support.")


if __name__ == "__main__":
    asyncio.run(main())
