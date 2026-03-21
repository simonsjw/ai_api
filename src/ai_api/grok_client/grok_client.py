#!/usr/bin/env python3
"""
Asynchronous client for xAI Grok API with batched requests and persistence.

This module provides an asynchronous HTTP client specifically tailored for interacting
with the xAI Grok API[](https://api.x.ai/v1/responses). It supports submitting batches
of requests in parallel with configurable concurrency, automatic retries on failures,
and optional persistence of responses. Persistence can be disabled ("none"), saved to
local JSON files ("json_files"), or inserted into a PostgreSQL database ("postgres")
using the custom `infopypg` library for connection pooling and query execution.

Key features:
- Batched asynchronous requests using `aiohttp` for high throughput.
- Retry logic with exponential backoff for transient errors (e.g., network issues,
  timeouts, or API rate limits).
- Persistence integration: JSON files for simple local storage or PostgreSQL for
  structured querying and scalability.
- Custom logging via the `logger` module, which supports both file-based and
  PostgreSQL logging (see `custom_module_use.md` for details).
- Type-safe request/response models using `dataclasses` for immutability and
  automatic generation of methods like `__init__` and `__repr__`.
- Includes functionality to set conv_id to aid with caching for cost reduction.
- Accepts output validation model using python dictionaries or pydantic.
  (Part of the custom GrokRequest model.)

Notes
-----
- This client uses the xAI Grok API endpoint /v1/responses, which differs from OpenAI's
  /chat/completions.
- For PostgreSQL mode, the `responses` table must exist or be creatable via `infopypg`
  (e.g., using `DatabaseBuilder` from `infopypg.setupdb`). The table schema should
  include columns like `endpoint` (JSONB), `response` (JSONB), and `note` (TEXT).
- The use of set_conv_id = True only reduces costs if prompts are not sent
  simultaneously. The cache can't be reused if all prompts are executed at the same
  time, before a cache can be created.
- The `@dataclass` decorator (from the `dataclasses` module) is used on `GrokRequest`
  and `GrokResponse`. It automatically generates special methods like `__init__`,
  `__repr__`, `__eq__`, and `__hash__` based on the class attributes, making the
  classes lightweight and immutable (with `frozen=True`). This reduces boilerplate
  code while ensuring type safety and readability. For example, it creates an
  `__init__` that accepts the specified fields as arguments.
- The `@overload` decorator (from `typing`) is used on `submit_batch` to provide
  multiple type signatures for the method. This allows type checkers (e.g., pyrefly)
  to infer different return types based on the `return_responses` parameter: a list
  of `GrokResponse` if True, or None if False. It's purely for static type analysis
  and does not affect runtime behaviour.
- Error handling: Logs failures via the custom logger and returns exceptions in
  results for caller inspection. Does not raise globally to allow partial successes
  in batches.
- Dependencies: Relies on `aiohttp` for async HTTP, `asyncpg` and `PgPoolManager` via
  `infopypg` for DB, and custom `logger` for logging.
  See `environment_grok.yml` for full package list.
- Initialisation uses an async factory method `create` to handle asynchronous setup
  tasks, such as resolving PostgreSQL connection settings and initialising the
  connection pool. This approach prioritises efficiency by ensuring non-blocking
  operations during client creation, which is essential in async contexts where
  blocking could stall the event loop. It draws from historical lessons in asynchronous
  programming (e.g., patterns in libraries like asyncpg and aiohttp), where factory
  methods separate synchronous attribute assignment in `__init__` from awaitable
  orchestration. This enhances clarity by keeping `__init__` lightweight and focused
  on state, while `create` manages complex async logic conditionally (e.g., only for
  "postgres" mode). For readability, users instantiate via
  `await XAIAsyncClient.create(...)`,
  making the async nature explicit and avoiding partially-initialised objects.
  Direct calls to `__init__` are discouraged, as they bypass async setup.

Parameters (for create)
------------------------
api_key : str
    The xAI API key for authentication (Bearer token).
save_mode : SaveMode, optional
    Persistence mode: "none" (default, no saving), "json_files" (local JSON),
    or "postgres" (DB insert).
output_dir : Path | None, optional
    Directory for JSON files if save_mode="json_files".
pg_settings : dict[str, Any] | ResolvedSettingsDict | None, optional
    PG settings (raw dict or resolved); resolved async in create if needed.
    Falls back to module-level default if None in "postgres" mode.
log_location : str | dict[str, Any] | ResolvedSettingsDict, optional
    Location for logger setup (file path or PG settings; default: "grok_client.log").
concurrency : int, optional
    Maximum concurrent requests (default: 50).
max_retries : int, optional
    Maximum retry attempts on failures (default: 3).
set_conv_id : bool | str, optional
    Add a conv_id UUID to each prompt to Grok. Since this id is set at initialisation,
    it will be the same for all prompts sent after the class is initialised. This allows
    Grok to attempt to use caching, potentially reducing costs. To have the value
    generated automatically, set to True. Alternatively, pass a string to set to
    that value.

Returns
-------
XAIAsyncClient
    The initialised client instance; use `submit_batch` for API interactions.

Raises
------
ValueError
    If save_mode configuration is invalid (e.g., missing settings/dir).
ConnectionError
    If PostgreSQL settings resolution or pool creation fails.
RuntimeError
    If unreachable code is hit (internal safeguard).

Examples
--------
 >>> import asyncio
 >>> from pathlib import Path
 >>> from ai_api.grok_client import XAIAsyncClient
 >>> from ai_api.data_structures import GrokRequest, GrokInput, GrokMessage
 >>> from infopypg.pgtypes import ResolvedSettingsDict

Example with JSON saving
 >>> async def main():
 ...     client = await XAIAsyncClient.create(
 ...         api_key="xai_ ...", save_mode="json_files", output_dir=Path("outputs")
 ...     )
 ...     messages = [{"role": "user", "content": "Hello, Grok!"}]
 ...     grok_input = GrokInput.from_list(messages)
 ...     req = GrokRequest(input=grok_input)
 ...     results = await client.submit_batch([req])
 ...     print(results[0].text)  # Prints the API response text
 >>> asyncio.run(main())

Example with PostgreSQL saving
 >>> async def main():
 ...     pg_settings: ResolvedSettingsDict = (
 ...         {"DB_USER": "simon", "DB_HOST": "localhost",  ...}
 ...     )
 ...     client_pg = await XAIAsyncClient.create(
 ...         api_key="xai_ ...", save_mode="postgres", pg_settings=pg_settings
 ...     )
 ...     results_pg = await client_pg.submit_batch([req], return_responses=False)
 ...     # None, responses saved to DB
 >>> asyncio.run(main())

Example with structured output validation
 >>> async def main():
 ...     client = await XAIAsyncClient.create(api_key="xai_ ...")
 ...     from pydantic import BaseModel
 ...
 ...     class MathResponse(BaseModel):
 ...         result: int
 ...         explanation: str
 ...
 ...     messages = [{"role": "user", "content": "What is 2 + 2?"}]
 ...     grok_input = GrokInput.from_list(messages)
 ...     req_struct = GrokRequest(input=grok_input, structured_schema=MathResponse)
 ...     results = await client.submit_batch([req_struct])
 ...     import json
 ...
 ...     parsed = json.loads(
 ...         results[0].text
 ...     )  # Or MathResponse.model_validate_json(results[0].text)
 >>> asyncio.run(main())

Example of sequential chaining for caching
 >>> async def main():
 ...     client = await XAIAsyncClient.create(
 ...         api_key="xai_ ...",
 ...         save_mode="postgres",  # Or your preferred mode
 ...         set_conv_id=True,  # Enables shared conv_id for caching
 ...     )
 ...     # Similar prompts (shared prefix for cache hit)
 ...     base_msg = "Analyse the following text: "
 ...     prompts = [
 ...         base_msg + "The quick brown fox jumps over the lazy dog.",
 ...         base_msg + "The quick brown fox jumps over the lazy cat.",
 ...         base_msg + "The quick brown fox jumps over the lazy rabbit.",
 ...     ]
 ...     responses = []
 ...     for prompt_text in prompts:
 ...         messages = [{"role": "user", "content": prompt_text}]
 ...         grok_input = GrokInput.from_list(messages)
 ...         req = GrokRequest(input=grok_input, model="grok-beta")
 ...         result = await client.submit_batch([req])  # Sequential await
 ...         if result:
 ...             responses.append(result[0])
 ...             print(f"Response: {result[0].text}")
 >>> asyncio.run(main())

"""

import asyncio
import json                                                                               # For dict serialisation to str (asyncpg binding with ::jsonb casts).
import os
import traceback
import uuid
from datetime import (                                                                    # Last 2 for partition date handling.
    UTC,
    datetime,
)
from pathlib import Path
from typing import Any, Literal, cast, overload

import aiohttp
from asyncpg import Pool
from dotenv import load_dotenv
from infopypg import (
    PgPoolManager,
    ResolvedSettingsDict,
    SettingsDict,
    async_dict_to_ResolvedSettingsDict,
    async_resolve_SettingsDict_to_ResolvedSettingsDict,
    ensure_partition_exists,
    execute_query,
    is_ResolvedSettingsDict,
    validate_dict_to_SettingsDict,
)
from logger import Logger, setup_logger

from ai_api.data_structures import (
    GrokRequest,
    GrokResponse,
    SaveMode,
)

_ = load_dotenv()

POSTGRES_DB_RESPONSES = (
    "{ "
    '"db_user":"postgres", "db_host":"127.0.0.1", '
    '"db_port":"5432", "db_name":"responsesdb", '
    '"password":"mgrr8X3OrEh", "tablespace_name":"responses_db", '
    '"tablespace_path":'
    '"/mnt/HDD03_HIT_03TB/no_backup/pg03/responses_db", '
    '"extensions":["uuid-ossp","pg_trgm"]'
    " }"
)


if not POSTGRES_DB_RESPONSES:                                                             # Check for presence early to avoid unnecessary work
    raise ValueError("POSTGRES_DB_RESPONSES environment variable is not set.")

try:
    responsesdb_settings: dict[str, str | list[str]] = json.loads(POSTGRES_DB_RESPONSES)
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in POSTGRES_DB_RESPONSES: {e}") from e


err_string: str


script_dir: str = os.path.dirname(os.path.abspath(__file__))
log_path: str = os.path.normpath(
    os.path.join(script_dir, "..", "..", "log", "default.log")
)


class XAIAsyncClient:
    """
    Asynchronous client for xAI Grok API with batched requests and persistence.

    Use async classmethod create for instantiation to handle async setup.
    """

    base_url: str = "https://api.x.ai/v1/responses"                                       # Class var: efficient constant
    timeout: float = 60.0                                                                 # Default timeout (secs); override in init if needed

    def __init__(
        self,
        api_key: str,
        save_mode: SaveMode = "none",
        output_dir: Path | None = None,
        pg_settings: dict[str, str | list[str]]
        | ResolvedSettingsDict = responsesdb_settings,
        log_location: str | dict[str, Any] | ResolvedSettingsDict = "grok_client.log",
        concurrency: int = 50,
        max_retries: int = 3,
        timeout: float = 60.0,
        set_conv_id: bool | str | None = False,
    ) -> None:
        """
        Sync init: Store args; async setup deferred to create.

        Parameters as per class docstring.
        """
        self.api_key = api_key
        self.save_mode = save_mode
        self.output_dir = output_dir
        self.pg_settings = pg_settings
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout = timeout
        self._set_conv_id = set_conv_id                                                   # Initialise private attr for property caching
        self.pool = None                                                                  # Deferred to create
        self.provider_id = 0                                                              # Internal sentinel; not a param
        self.logger: Logger = setup_logger("GROK_CLIENT", log_location=log_location)
        self._log_location = log_location                                                 # stored for any value not in

        clean_pg_settings: dict[str, str | list[str]] = {
            k: v for k, v in cast(dict, pg_settings).items() if k.upper() != "PASSWORD"
        }

        if is_ResolvedSettingsDict(log_location) or type(log_location) == dict:
            clean_log_location: str | dict[str, str | list[str]] = {
                k: v
                for k, v in cast(dict, log_location).items()
                if k.upper() != "PASSWORD"
            }
        else:
            clean_log_location = cast(str, log_location)

        self._initial_args = {
            "pg_settings": clean_pg_settings,
            "save_mode": self.save_mode,
            "output_dir": self.output_dir,
            "log_location": clean_log_location,
            "concurrency": self.concurrency,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "set_conv_id": self._set_conv_id,
        }

    @classmethod
    async def create(
        cls,
        api_key: str,
        save_mode: SaveMode = "none",
        output_dir: Path | None = None,
        pg_settings: dict[str, str | list[str]]
        | ResolvedSettingsDict = responsesdb_settings,
        log_location: str | dict[str, Any] | ResolvedSettingsDict = "grok_client.log",
        concurrency: int = 50,
        max_retries: int = 3,
        set_conv_id: bool = False,
    ) -> "XAIAsyncClient":
        """
        Async factory: Create instance and perform async setup.

        Flow:
        1. Instantiate via __init__ (sync).
        2. If postgres, await resolution of pg_settings to resolved_pg_settings.
        3. Await pool init if needed.
        4. Return instance.

        Parameters
        ----------
        pg_settings : Dict[str, Any] | ResolvedSettingsDict | None, optional
            Raw PG settings dict or pre-resolved; resolved async if raw.

        # ... (other params unchanged)

        Returns
        -------
        XAIAsyncClient
            Ready client.

        Raises
        ------
        ValueError
            Invalid settings.
        ConnectionError
            Resolution/pool failure.
        """
        instance = cls(
            api_key=api_key,
            save_mode=save_mode,
            output_dir=output_dir,
            pg_settings=pg_settings,
            log_location=log_location,
            concurrency=concurrency,
            max_retries=max_retries,
            set_conv_id=set_conv_id,
        )

        if instance.save_mode == "postgres" and instance.pg_settings:
            if not is_ResolvedSettingsDict(instance.pg_settings):
                instance.pg_settings = await async_dict_to_ResolvedSettingsDict(
                    cast(dict[str, str | list[str]], instance.pg_settings)
                )

            instance.pool = await PgPoolManager.get_pool(instance.pg_settings)

        return instance

    @property
    def conv_id(self) -> str | None:
        """
        Lazily resolve and return the conversation ID.

        If _set_conv_id is a string, return it directly.
        If True, generate and cache a UUID v4 string on first access.
        Otherwise, return None.

        This property ensures efficiency by generating the UUID only when accessed,
        avoiding overhead in __init__ if conv_id is never used. Subsequent accesses
        return the cached value without regeneration.
        """
        if isinstance(self._set_conv_id, str):
            return self._set_conv_id                                                      # Direct use if provided as string.
        elif self._set_conv_id is not None:
            # Generate and cache if configured for auto-generation.
            self._set_conv_id = str(uuid.uuid4())
            return self._set_conv_id
        else:
            return None                                                                   # Default to None if not enabled.

    async def _get_provider_id(self) -> int:
        """
        Retrieve or insert the provider ID for 'GROK'.

        Queries the providers table for name='GROK'. If not found, inserts it and
        returns the ID.

        Caches the result in self.provider_id for subsequent calls.

        Returns
        -------
        int
            The provider ID.

        Raises
        ------
        asyncpg.PostgresError
            On query/insert failure.
        """
        if self.provider_id != 0:
            return self.provider_id

        connection_pool: Pool = await self._get_pool()

        result = await execute_query(
            connection_pool,
            query_sql="SELECT id FROM public.providers WHERE name = $1",
            params=["GROK"],
            fetch=True,
        )
        if not result:
            result = await execute_query(
                connection_pool,
                query_sql=(
                    "INSERT INTO public.providers (name) " "VALUES ($1) " "RETURNING id"
                ),
                params=["GROK"],
                fetch=True,
            )
            self.logger.info(msg="Inserted 'Grok' into providers table.")
        if result:
            self.provider_id = cast(int, result[0]["id"])
            return self.provider_id

        raise RuntimeError("Failed to get or insert provider ID for 'GROK'.")

    def _filename_for(self, resp_id: str) -> Path:
        """
        Generate a timestamped filename for JSON saving.

        Parameters
        ----------
        resp_id : str
            Response ID from API.

        Returns
        -------
        Path
            Full path to the JSON file.
        """
        ts = datetime.now(UTC).strftime(format="%Y%m%dT%H%M%SZ")
        return self.output_dir / f"{ts}_{resp_id}.json"                                   # type: ignore[arg-type]

    async def _get_pool(self) -> Pool:
        """
        Get the connection pool, leveraging instance settings. If a SettingsDict has
        been provided but not processed, it will be processed here to produce the
        ResolvedSettingsDict needed for the use of PgPoolManager.get_pool.

        Returns
        -------
        Pool
            The cached pool.

        Raises
        ------
        ValueError
            If not in postgres mode or settings missing.
        """
        if self.save_mode != "postgres" or self.pg_settings is None:
            self.logger.error("No DB settings available.")
            raise ValueError("DB settings required for PostgreSQL operations.")

        if not is_ResolvedSettingsDict(self.pg_settings):
            self.pg_settings = await async_dict_to_ResolvedSettingsDict(
                cast(dict[str, str | list[str]], self.pg_settings)
            )

        return await PgPoolManager.get_pool(
            self.pg_settings
        )                                                                                 # Reuse instance resolved settings

    async def _save_request_to_postgres(
        self, headers: dict[str, str], req: GrokRequest
    ) -> tuple[uuid.UUID, datetime]:
        """
        Insert the request into PostgreSQL and return the request_id and tstamp.

        Generates a new UUID for request_id since it's not part of GrokRequest.
        Uses req.to_dict() for the request payload.

        Parameters
        ----------
        headers : dict[str, str]
            The header component of the prompt.
        req : GrokRequest
            The immutable request to save.

        Returns
        -------
        tuple[uuid.UUID, datetime]
            The generated request_id and inserted tstamp.

        Raises
        ------
        asyncpg.PostgresError
            On query execution failure (handled internally by execute_query).
        RuntimeError
            If insert succeeds but no tstamp is returned (safeguard).
        """
        request_id: uuid.UUID = uuid.uuid4()                                              # Always generate since not in req

        provider_id = await self._get_provider_id()
        endpoint_json = {
            "ORG": "GROK",
            "MODEL": req.model,
            "ENDPOINT": self.base_url,
        }
        request_json: dict = req.to_dict()                                                # Use built-in serialization

        # Get the settings for the Grock client.
        meta_args = self._initial_args

        # Serialise dicts to str for binding.
        initial_args_str: str = json.dumps({**meta_args, "headers": headers})
        endpoint_str: str = json.dumps(endpoint_json)
        request_str: str = json.dumps(request_json)

        connection_pool: Pool = await self._get_pool()
        tstamp: datetime = datetime.now(UTC)

        # Ensure partition for today
        await ensure_partition_exists(
            connection_pool,
            table_name="requests",
            target_date=None,                                                             # Defaults to today.
            range_interval="daily",
            look_ahead_days=2,
        )

        result = await execute_query(
            connection_pool,
            query_sql=(
                "INSERT INTO public.requests (tstamp, provider_id, endpoint, "
                "request_id, request, meta) "
                "VALUES ($1, $2, $3::jsonb, $4, $5::jsonb, $6::jsonb) "
                "RETURNING tstamp"
            ),
            params=[
                tstamp,
                provider_id,
                endpoint_str,
                request_id,
                request_str,
                initial_args_str,
            ],
            fetch=True,
        )
        if not result:
            raise RuntimeError("Failed to insert request and retrieve tstamp.")

        inserted_tstamp: datetime = result[0]["tstamp"]
        self.logger.info(
            msg=f"Inserted request {request_id} into PostgreSQL.",
            extra={"obj": {"request_id": str(request_id)}},
        )
        return request_id, inserted_tstamp

    async def _save_response_to_postgres(
        self,
        meta: dict[str, dict[str, str] | float],
        grok_resp: GrokResponse,
        request_id: uuid.UUID,
        request_tstamp: datetime,
    ) -> None:
        """
        Insert the response into PostgreSQL, linking to the request via request_id and
        request_tstamp.

        Parameters
        ----------
        meta : dict[str, dict[str, str] | float]
            Metadata relating to the outgoing request for the response.
        grok_resp : GrokResponse
            The response to save.
        request_id : uuid.UUID
            The linked request_id.
        request_tstamp : datetime
            The linked request's tstamp.

        Raises
        ------
        asyncpg.PostgresError
            On insertion failure (handled by execute_query).
        """
        provider_id = await self._get_provider_id()
        endpoint_json = {
            "ORG": "GROK",
            "MODEL": grok_resp.model,
            "ENDPOINT": self.base_url,
        }

        # Serialise dicts to str for binding.
        headers_str: str = json.dumps(meta)
        endpoint_str: str = json.dumps(endpoint_json)
        response_str: str = json.dumps(grok_resp.raw)

        connection_pool: Pool = await self._get_pool()
        tstamp: datetime = datetime.now(UTC)

        # Ensure partition for today
        await ensure_partition_exists(
            connection_pool,
            table_name="responses",
            target_date=None,                                                             # Defaults to today.
            range_interval="daily",
            look_ahead_days=2,
        )

        await execute_query(
            connection_pool,
            query_sql=(
                "INSERT INTO public.responses (tstamp, provider_id, endpoint, "
                "request_id, request_tstamp, "
                "response_id, response, meta) "
                "VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7::jsonb, $8::jsonb)"
            ),
            params=[
                tstamp,
                provider_id,
                endpoint_str,
                request_id,
                request_tstamp,
                grok_resp.id,                                                             # str ID as per new GrokResponse
                response_str,
                headers_str,
            ],
            fetch=False,
        )
        self.logger.info(
            msg=f"Inserted response {grok_resp.id} into PostgreSQL.",
            extra={"obj": {"response_id": grok_resp.id}},
        )

    async def _single_call(
        self,
        session: aiohttp.ClientSession,
        req: GrokRequest,
        semaphore: asyncio.Semaphore,
    ) -> GrokResponse | Exception:
        async with semaphore:
            request_id = None
            request_tstamp = None

            if self.conv_id:                                                              # Reuse same ID for related requests
                headers: dict[str, str] = {
                    "Authorization": f"Bearer {self.api_key}",
                    "x-grok-conv-id": self.conv_id,
                }
            else:
                headers = {"Authorization": f"Bearer {self.api_key}"}

            if self.save_mode == "postgres":
                try:
                    request_id, request_tstamp = await self._save_request_to_postgres(
                        headers, req
                    )
                except Exception as e:
                    print(
                        "Save request failed with traceback:\n", traceback.format_exc()
                    )                                                                     # Add this
                    self.logger.error(
                        msg=f"Failed to save request: {e}",
                        extra={"obj": {"error": str(e)}},
                    )
                    # Continue or raise e if you want to halt

            payload = req.to_dict()

            for attempt in range(self.max_retries + 1):
                try:
                    async with session.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status != 200:
                            txt = await resp.text()
                            raise aiohttp.ClientError(f"HTTP {resp.status}: {txt}")

                        data = await resp.json()
                        grok_resp = GrokResponse.from_dict(data)

                        if self.save_mode == "json_files":
                            path: Path = self._filename_for(grok_resp.id)
                            path.write_text(
                                data=json.dumps(data, ensure_ascii=False, indent=2)
                            )
                            self.logger.info(
                                msg=f"Saved response to {path.name}.",
                                extra={"obj": {"file": str(path)}},
                            )

                        elif (
                            self.save_mode == "postgres"
                            and request_id
                            and request_tstamp
                        ):
                            await self._save_response_to_postgres(
                                {"headers": headers, "timeout": self.timeout},
                                grok_resp,
                                request_id,
                                request_tstamp,
                            )

                        return grok_resp

                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    json.JSONDecodeError,
                    Exception,
                ) as e:
                    if attempt == self.max_retries:
                        self.logger.error(
                            msg=f"Permanent failure: {e}",
                            extra={"obj": {"error": str(e)}},
                        )
                        return e
                    wait = 1.5**attempt
                    self.logger.warning(
                        msg=f"Retry {attempt + 1}/{self.max_retries}: {e}",
                        extra={"obj": {"attempt": attempt}},
                    )
                    await asyncio.sleep(wait)

            return RuntimeError("Unreachable")

    @overload
    async def submit_batch(self, requests: list[GrokRequest]) -> list[GrokResponse]: ...

    # Overload for default (return_responses=True): returns list.

    @overload
    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: Literal[True]
    ) -> list[GrokResponse]: ...

    # Explicit True: same as default, returns list.

    @overload
    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: Literal[False]
    ) -> None: ...

    # Explicit False: returns None.

    async def submit_batch(
        self, requests: list[GrokRequest], return_responses: bool = True
    ) -> list[GrokResponse] | None:
        """
        Submit a batch of requests in parallel and optionally return responses.

        This method uses asyncio.gather for concurrent execution, constrained by
        a semaphore to limit parallelism. It filters out exceptions (logging them)
        and collects successful responses. The flow is:
        1. Check for empty requests and early return.
        2. Create semaphore based on client concurrency limit.
        3. Open aiohttp session.
        4. Generate tasks for each request.
        5. Await gathered tasks, ignoring exceptions in results.
        6. Collect valid responses.
        7. Return responses if requested, else None.

        Parameters
        ----------
        requests : list[GrokRequest]
            List of API requests to process concurrently.
        return_responses : bool, optional
            Flag to return collected responses (default: True).

        Returns
        -------
        list[GrokResponse] | None
            List of successful responses if return_responses is True; else None.

        Raises
        ------
        No explicit raises; exceptions from tasks are logged, not propagated.

        Examples
        --------
        >>> await client.submit_batch([req1, req2])  # Returns [resp1, resp2]
        >>> await client.submit_batch([req1, req2], return_responses=False)  # Returns None
        """
        if not requests:
            return [] if return_responses else None

        semaphore = asyncio.Semaphore(self.concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = [self._single_call(session, req, semaphore) for req in requests]
            results: list[Exception | GrokResponse] = await asyncio.gather(
                *tasks, return_exceptions=False
            )

        responses: list[GrokResponse] = []
        for item in results:
            if isinstance(item, Exception):
                self.logger.error(
                    msg=f"Task failed: {item}", extra={"obj": {"error": str(item)}}
                )
            else:
                responses.append(item)

        return responses if return_responses else None
