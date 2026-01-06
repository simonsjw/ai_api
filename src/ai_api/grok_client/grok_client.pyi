from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, overload
import uuid

from ..data_structures import SaveMode, GrokResponse, GrokRequest
from infopypg import ResolvedSettingsDict
import aiohttp


class XAIAsyncClient:
    api_key: str
    base_url: str
    save_mode: SaveMode
    output_dir: Path | None
    concurrency: int
    timeout: aiohttp.ClientTimeout
    max_retries: int
    _resolved_pg_settings: ResolvedSettingsDict | None
    provider_id: int

    def __init__(
        self,
        api_key: str,
        *,
        save_mode: SaveMode = "none",
        resolved_pg_settings: ResolvedSettingsDict | None = None,
        output_dir: Path | None = None,
        concurrency: int = 50,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None: ...

    def _make_payload(self, req: GrokRequest) -> dict[str, Any]: ...

    def _filename_for(self, req: GrokRequest, resp_id: str) -> Path: ...

    async def _get_provider_id(self) -> int: ...

    async def _save_request_to_postgres(self, req: GrokRequest) -> tuple[uuid.UUID, datetime]: ...

    async def _save_response_to_postgres(
        self, resp: GrokResponse, request_id: uuid.UUID, request_tstamp: datetime
    ) -> None: ...

    async def _single_call(
        self,
        session: aiohttp.ClientSession,
        req: GrokRequest,
        semaphore: asyncio.Semaphore,
    ) -> GrokResponse | Exception: ...

    @overload
    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: Literal[True]
    ) -> list[GrokResponse]: ...

    @overload
    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: Literal[False]
    ) -> None: ...

    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: bool = True
    ) -> list[GrokResponse] | None: ...
