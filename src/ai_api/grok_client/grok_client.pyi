from pathlib import Path
from typing import Literal, overload

import aiohttp
from infopypg import ResolvedSettingsDict as ResolvedSettingsDict
from logger import Logger

from ai_api.data_structures import (
    GrokRequest as GrokRequest,
)
from ai_api.data_structures import (
    GrokResponse as GrokResponse,
)
from ai_api.data_structures import (
    SaveMode as SaveMode,
)
from ai_api.data_structures import (
    responses_default_settings as responses_default_settings,
)

err_string: str
logger: Logger

class XAIAsyncClient:
    api_key: str
    base_url: str
    save_mode: SaveMode
    output_dir: Path | None
    concurrency: int
    timeout: aiohttp.ClientTimeout
    max_retries: int
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
        set_conv_id: str | bool = False,
    ) -> None: ...
    @property
    def conv_id(self) -> str | None: ...
    @overload
    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: Literal[True]
    ) -> list[GrokResponse]: ...
    @overload
    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: Literal[False]
    ) -> None: ...
