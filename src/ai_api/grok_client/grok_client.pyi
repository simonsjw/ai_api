from pathlib import Path
from typing import Any, Literal, overload

from _typeshed import Incomplete
from infopypg import ResolvedSettingsDict as ResolvedSettingsDict
from infopypg import SettingsDict as SettingsDict
from logger import Logger

from ai_api.data_structures import GrokRequest as GrokRequest
from ai_api.data_structures import GrokResponse as GrokResponse
from ai_api.data_structures import SaveMode as SaveMode

_: Incomplete
POSTGRES_DB_RESPONSES: str
POSTGRES_DB_MEMES: str
responsesdb_settings: dict[str, str | list[str]]
responsesdb_validated_settings: SettingsDict
err_string: str
script_dir: str
log_path: str

class XAIAsyncClient:
    api_key: Incomplete
    save_mode: Incomplete
    output_dir: Incomplete
    resolved_pg_settings: Incomplete
    concurrency: Incomplete
    max_retries: Incomplete
    set_conv_id: Incomplete
    pool: Incomplete
    logger: Logger
    def __init__(
        self,
        api_key: str,
        save_mode: SaveMode = "none",
        output_dir: Path | None = None,
        resolved_pg_settings: dict[str, Any] | ResolvedSettingsDict | None = None,
        log_location: str | dict[str, Any] | ResolvedSettingsDict = "grok_client.log",
        concurrency: int = 50,
        max_retries: int = 3,
        set_conv_id: bool = False,
    ) -> None: ...
    @classmethod
    async def create(
        cls,
        api_key: str,
        save_mode: SaveMode = "none",
        output_dir: Path | None = None,
        resolved_pg_settings: dict[str, Any] | ResolvedSettingsDict | None = None,
        log_location: str | dict[str, Any] | ResolvedSettingsDict = "grok_client.log",
        concurrency: int = 50,
        max_retries: int = 3,
        set_conv_id: bool = False,
    ) -> XAIAsyncClient: ...
    @property
    def conv_id(self) -> str | None: ...
    @overload
    async def submit_batch(self, requests: list[GrokRequest]) -> list[GrokResponse]: ...
    @overload
    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: Literal[True]
    ) -> list[GrokResponse]: ...
    @overload
    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: Literal[False]
    ) -> None: ...
