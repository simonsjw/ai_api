#!/usr/bin/env python3
"""
Script to define database schema models using SQLAlchemy.

This script contains the declarative base and model classes for tables.
Models define columns with types, defaults, and constraints. These are
used by the `setup_db.py` script to create tables via metadata.

Notes
-----
- Uses infopypg for setup; assumes it's installed via pip -e.
- id is BIGINT IDENTITY (unique across partitions via composite PK with tstamp).
- Refinements: Normalized providers (few unique: 5-10) with lookup.
  Partition Responses by tstamp (RANGE, daily suggested) for growth.
  Sub-partitions must be created post-setup (e.g., via SQL function).
  Index on tstamp for sorting/queries. JSONB fields as dict[str, Any] for flexibility.
- Extensions: Trimmed to essentials; add more via env if needed.
- For daily aggregates: Use queries like SELECT date_trunc('day', tstamp), COUNT(*) ...
  Push down to PG for efficiency.

Flow:
- Define default_settings (SettingsDict) with env overrides.
- Define Base for ORM inheritance.
- Define tables as classes with mapped columns.

Parameters
----------
None
    This is a spec file; imported by DatabaseBuilder.

Returns
-------
None
    Exports models and settings for setupdb.

Raises
------
ImportError
    If infopypg or deps missing; install via conda env.
ValueError
    If env vars invalid; check POSTGRES_*.

Examples
--------
>>> from infopypg import DatabaseBuilder, resolve_postgres_connection_settings
>>> resolved = resolve_postgres_connection_settings(default_settings=default_settings)
>>> builder = DatabaseBuilder(resolved_settings=resolved)
>>> await builder.build()  # In async context; creates partitioned table (no sub-partitions)
"""

import asyncio
import uuid
from datetime import datetime
from os import getenv
from typing import Any

from infopypg import (
    Base,
    DatabaseBuilder,
)
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
from sqlalchemy.dialects.postgresql import (
    JSONB,
)
from sqlalchemy.dialects.postgresql import (
    UUID as pgUUID,
)
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

    Use FK from Responses for efficiency; avoids storing repeated strings.
    No partitioning needed (small table).
    """

    __tablename__: str = "providers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )                                                                                     # Optional details.

    __table_args__: tuple[Index, dict[str, bool]] = (
        Index("ix_providers_name", "name"),
        {"extend_existing": True},
    )


class Requests(Base):
    """
    Main table for LLM requests (partitioned by tstamp for growth).

    Refinements:
    - id: BIGINT IDENTITY (autoincrement per partition); composite PK (id, tstamp) for uniqueness.
    - provider_id: FK to Providers (normalized).
    - Partitioning: RANGE by tstamp (daily aggregates via queries).
      Sub-partitions: Create post-setup, e.g., FOR VALUES FROM ('YYYY-MM-DD') TO ('YYYY-MM-DD+1').
    - Sorting: Index on tstamp for ORDER BY.
    - JSONB: dict[str, Any] for endpoint/meta (structured tags/notes); request as dict for flexibility.
    """

    __tablename__: str = "requests"

    idx: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),                                                            # Per-partition autoincrement.
        primary_key=True,
    )
    tstamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), primary_key=True
    )                                                                                     # Partition key; part of PK.
    provider_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("providers.id"), nullable=False
    )
    endpoint: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    request_id: Mapped[uuid.UUID] = mapped_column(pgUUID, nullable=False)
    request: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    __table_args__: tuple[
        Index, Index, Index, UniqueConstraint, dict[str, bool | str]
    ] = (
        Index("ix_requests_tstamp", "tstamp"),
        Index("ix_requests_provider_id", "provider_id"),
        Index(
            "ix_requests_request_id", "request_id"
        ),                                                                                # Added for faster lookups/joins on request_id
        UniqueConstraint(
            "request_id", "tstamp", name="uq_requests_request_id_tstamp"
        ),                                                                                # Required for composite FK reference
        {"postgresql_partition_by": "RANGE (tstamp)", "extend_existing": True},
    )


class Responses(Base):
    """
    Main table for LLM responses (partitioned by tstamp for growth).

    Refinements:
    - id: BIGINT IDENTITY (autoincrement per partition); composite PK (id, tstamp) for uniqueness.
    - provider_id: FK to Providers (normalized).
    - Partitioning: RANGE by tstamp (daily aggregates via queries).
      Sub-partitions: Create post-setup, e.g., FOR VALUES FROM ('YYYY-MM-DD') TO ('YYYY-MM-DD+1').
    - Sorting: Index on tstamp for ORDER BY.
    - JSONB: dict[str, Any] for endpoint/meta (structured tags/notes); response as dict for flexibility.
    - Foreign Key: Composite FK (request_id, request_tstamp) references requests (request_id, tstamp) for 1:1 referential integrity.
    """

    __tablename__: str = "responses"

    idx: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),                                                            # Per-partition autoincrement.
        primary_key=True,
    )
    tstamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), primary_key=True
    )                                                                                     # Partition key; part of PK.
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

    __table_args__: tuple[
        Index, Index, Index, Index, Index, ForeignKeyConstraint, dict[str, bool | str]
    ] = (
        Index("ix_responses_tstamp", "tstamp"),
        Index("ix_responses_provider_id", "provider_id"),
        Index(
            "ix_responses_request_id", "request_id"
        ),                                                                                # Added for faster joins on request_id
        Index(
            "ix_responses_response_id", "response_id"
        ),                                                                                # Added for faster lookups on response_id
        Index(
            "ix_responses_request_id_tstamp", "request_id", "request_tstamp"
        ),                                                                                # For composite FK efficiency
        ForeignKeyConstraint(
            columns=["request_id", "request_tstamp"],
            refcolumns=["requests.request_id", "requests.tstamp"],
            name="fk_responses_request_id_tstamp",
            ondelete="CASCADE",                                                           # CASCADE deletes response if request is deleted
        ),
        {"postgresql_partition_by": "RANGE (tstamp)", "extend_existing": True},
    )


class Logs(Base):
    """
    Table for logging records (partitioned by tstamp for growth).

    Refinements:
    - idx: BIGINT IDENTITY (autoincrement per partition); composite PK (idx, tstamp) for uniqueness.
    - Partitioning: RANGE by tstamp (daily aggregates via queries).
      Sub-partitions: Create post-setup, e.g., FOR VALUES FROM ('YYYY-MM-DD') TO ('YYYY-MM-DD+1').
    - Sorting: Index on tstamp for ORDER BY.
    - JSONB: dict[str, Any] for obj (structured extra data); nullable for flexibility.
    """

    __tablename__: str = "logs"

    idx: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),                                                            # Per-partition autoincrement.
        primary_key=True,
    )
    tstamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), primary_key=True
    )                                                                                     # Partition key; part of PK.
    loglvl: Mapped[str] = mapped_column(Text, nullable=False)
    logger: Mapped[str] = mapped_column(Text, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    obj: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__: tuple[Index, dict[str, str | bool]] = (
        Index("ix_logs_tstamp", "tstamp"),
        {"postgresql_partition_by": "RANGE (tstamp)", "extend_existing": True},
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
        "extensions": [
            "uuid-ossp",                                                                  # For potential UUIDs.
            "postgres_fdw" "pg_trgm",                                                     # For text search if needed.
        ],
    }
    # settings_dict: SettingsDict = validate_dict_to_SettingsDict(settings)
    builder: DatabaseBuilder = DatabaseBuilder(settings)
    await builder.build()

    print("Database initialised.")


if __name__ == "__main__":
    asyncio.run(main())
