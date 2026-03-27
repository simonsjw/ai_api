import uuid
from datetime import datetime as datetime
from infopypg import Base
from sqlalchemy import ForeignKeyConstraint, Index, UniqueConstraint
from sqlalchemy.orm import Mapped
from typing import Any

raw_ext: str | None
extensions: list[str]

class Providers(Base):
    __tablename__: str
    id: Mapped[int]
    name: Mapped[str]
    description: Mapped[str | None]
    __table_args__: tuple[Index, dict[str, bool]]

class Requests(Base):
    __tablename__: str
    idx: Mapped[int]
    tstamp: Mapped[datetime]
    provider_id: Mapped[int]
    endpoint: Mapped[dict[str, Any]]
    request_id: Mapped[uuid.UUID]
    request: Mapped[dict[str, Any]]
    meta: Mapped[dict[str, Any]]
    __table_args__: tuple[Index, Index, Index, UniqueConstraint, dict[str, bool | str]]

class Responses(Base):
    __tablename__: str
    idx: Mapped[int]
    tstamp: Mapped[datetime]
    provider_id: Mapped[int]
    endpoint: Mapped[dict[str, Any]]
    request_id: Mapped[uuid.UUID]
    request_tstamp: Mapped[datetime]
    response_id: Mapped[uuid.UUID]
    response: Mapped[dict[str, Any]]
    meta: Mapped[dict[str, Any]]
    __table_args__: tuple[Index, Index, Index, Index, Index, ForeignKeyConstraint, dict[str, bool | str]]

class Logs(Base):
    __tablename__: str
    idx: Mapped[int]
    tstamp: Mapped[datetime]
    loglvl: Mapped[str]
    logger: Mapped[str]
    message: Mapped[str]
    obj: Mapped[dict[str, Any] | None]
    __table_args__: tuple[Index, dict[str, str | bool]]

async def main() -> None: ...
