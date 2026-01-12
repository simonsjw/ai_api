#!/usr/bin/env python3
"""
Asynchronous client for xAI Grok API with batched requests and persistence.

This module provides an asynchronous HTTP client specifically tailored for interacting
with the xAI Grok API[](https://api.x.ai/v1/chat/completions). It supports submitting
batches of requests in parallel with configurable concurrency, automatic retries on
failures, and optional persistence of responses. Persistence can be disabled ("none"),
saved to local JSON files ("json_files"), or inserted into a PostgreSQL database
("postgres") using the custom `infopypg` library for connection pooling and query
execution.

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
- This client assumes the xAI API follows OpenAI-compatible chat completion endpoints.
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

Parameters (for XAIAsyncClient init)
------------------------------------
api_key : str
    The xAI API key for authentication (Bearer token).
save_mode : SaveMode, optional
    Persistence mode: "none" (default, no saving), "json_files" (local JSON),
    or "postgres" (DB insert).
output_dir : Path | None, optional
    Directory for JSON files if save_mode="json_files".
resolved_pg_settings : ResolvedSettingsDict | None, optional
    Validated PostgreSQL settings if save_mode="postgres" (from `infopypg.pgtypes`).
concurrency : int, optional
    Maximum concurrent requests (default: 50).
timeout : float, optional
    Total request timeout in seconds (default: 60.0).
max_retries : int, optional
    Maximum retry attempts on failures (default: 3).
set_conv_id : bool, optional
    Add a conv_id UUID to each prompt to Grok. Since this id is set at initialisation,
    it will be the same for all prompts sent after the class is initialised. This allows
    Grok to attempt to use caching, potentially reducing costs. To have the value
    generated automatically, set to True. Alternatively, pass a string to set to
    that value.

Returns
-------
None
    Initialises the client; use `submit_batch` for API interactions.

Raises
------
ValueError
    If save_mode configuration is invalid (e.g., missing settings/dir).
RuntimeError
    If unreachable code is hit (internal safeguard).

Examples
--------
>>> import asyncio
>>> from pathlib import Path
>>> from ai_api.grok_client import XAIAsyncClient
>>> from ai_api.data_structures import GrokRequest, GrokMessage
>>> from infopypg.pgtypes import ResolvedSettingsDict

>>> # Example with JSON saving
>>> client = XAIAsyncClient(
...     api_key="xai_...", save_mode="json_files", output_dir=Path("outputs")
... )
>>> req = GrokRequest(messages=[GrokMessage(role="user", content="Hello, Grok!")])
>>> results = asyncio.run(client.submit_batch([req]))
>>> print(results[0].content)  # Prints the API response content

>>> # Example with PostgreSQL saving
>>> pg_settings: ResolvedSettingsDict = {"DB_USER": "simon", "DB_HOST": "localhost", ...}
>>> client_pg = XAIAsyncClient(
...     api_key="xai_...", save_mode="postgres", resolved_pg_settings=pg_settings
... )
>>> results_pg = asyncio.run(
...     client_pg.submit_batch([req], return_responses=False)
... )  # None, responses saved to DB

>>> # Example with structured output validation
>>> from pydantic import BaseModel
>>> class MathResponse(BaseModel):
...     result: int
...     explanation: str
>>> req_struct = GrokRequest(
...     messages=[GrokMessage(role="user", content="What is 2 + 2?")],
...     structured_schema=MathResponse,  # Or dict schema: {"type": "object", "properties": {...}}
... )
>>> results = asyncio.run(client.submit_batch([req_struct]))
>>> import json
>>> parsed = json.loads(
...     results[0].content
... )  # Or MathResponse.model_validate_json(results[0].content)

>>> # Example of sequential chaining for caching
>>> async def main():
... client = XAIAsyncClient(
...     api_key="xai_...",
...     save_mode="postgres",  # Or your preferred mode
...     set_conv_id=True,  # Enables shared conv_id for caching
... )
... # Similar prompts (shared prefix for cache hit)
>>> base_msg = "Analyse the following text: "
>>> prompts = [
...     base_msg + "The quick brown fox jumps over the lazy dog.",
...     base_msg + "The quick brown fox jumps over the lazy cat.",
...     base_msg + "The quick brown fox jumps over the lazy rabbit.",
... ]
>>> responses = []
>>> for prompt_text in prompts:
...     messages = [GrokMessage(role="user", content=prompt_text)]
...     req = GrokRequest(messages=messages, model="grok-beta")
...     result = await client.submit_batch([req])  # Sequential await
...     if result:
...         responses.append(result[0])
...         print(f"Response: {result[0].content}")
>>> asyncio.run(main())

"""

import asyncio
import json
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Type, cast, overload

import aiohttp
from asyncpg import Pool
from infopypg import (
    PgPoolManager,
    ResolvedSettingsDict,
    execute_query,
)
from logger import Logger, setup_logger
from pydantic import BaseModel

from ai_api.data_structures import (
    GrokRequest,
    GrokResponse,
    SaveMode,
    responses_default_settings,
)

err_string: str

logger: Logger = setup_logger(
    logger_name="XAI", log_location=responses_default_settings
)

logger.info("Initialising XAIAsyncClient with infopypg integration.")


class XAIAsyncClient:
    """
    Asynchronous client for batched xAI Grok API interactions with persistence.

    See module docstring for full details.
    """

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
    ) -> None:
        self.api_key: str = api_key
        self.base_url: str = "https://api.x.ai/v1/responses"
        self.save_mode: SaveMode = save_mode
        self.output_dir: Path | None = output_dir
        self.concurrency: int = concurrency
        self.timeout: aiohttp.ClientTimeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries: int = max_retries
        self._resolved_pg_settings: ResolvedSettingsDict | None = resolved_pg_settings
        self.provider_id: int = 0
        self._set_conv_id = set_conv_id

        if save_mode == "postgres":
            if resolved_pg_settings is None:
                err_string = "resolved_pg_settings required for 'postgres' mode."
                logger.error(err_string)
                raise ValueError(err_string)
            else:
                self._resolved_pg_settings = resolved_pg_settings

        elif save_mode == "json_files" and (not output_dir or not output_dir.is_dir()):
            err_string = "output_dir must be a valid directory for 'json_files' mode."
            logger.error(msg=err_string)
            raise ValueError(err_string)

        elif save_mode == "none":
            err_string = "Functionality not yet defined."
            print(err_string)
            raise ValueError(err_string)
        else:
            err_string = (
                "save_mode not recognised. Options are: json_files or postgres."
            )
            logger.error(msg=err_string)
            raise ValueError(err_string)

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
        elif self._set_conv_id is True:
            # Generate and cache if configured for auto-generation.
            self._set_conv_id = str(uuid.uuid4())
            return self._set_conv_id
        else:
            return None                                                                   # Default to None if not enabled.

    async def _get_provider_id(self) -> int:
        """
        Retrieve or insert the provider ID for 'GROK'.

        Queries the providers table for name='GROK'. If not found, inserts it and returns the ID.
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
            query_sql="SELECT id FROM providers WHERE name = $1",
            params=["GROK"],
            fetch=True,
        )
        if not result:
            result = await execute_query(
                connection_pool,
                query_sql="INSERT INTO providers (name) VALUES ($1) RETURNING id",
                params=["GROK"],
                fetch=True,
            )
            logger.info(msg="Inserted 'Grok' into providers table.")
        if result:
            self.provider_id = cast(int, result[0]["id"])
            return self.provider_id

        raise RuntimeError("Failed to get or insert provider ID for 'GROK'.")

    def _make_payload(self, req: GrokRequest) -> dict[str, Any]:
        """
        Construct the JSON payload for a Grok API request, including structured outputs if specified.

        If structured_schema is a dict, uses it directly as the schema.
        If it's a Pydantic BaseModel, generates the schema efficiently via model_json_schema().
        Prioritises efficiency by avoiding unnecessary imports or computations.

        Parameters
        ----------
        req : GrokRequest
            The request object.

        Returns
        -------
        dict[str, Any]
            API-compatible payload.

        Raises
        ------
        ValueError
            If structured_schema is invalid or Pydantic is missing when needed.
        """
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": [
                msg.to_dict() for msg in req.messages
            ],                                                                            # Assuming GrokMessage has to_dict(); adjust if needed
            "temperature": req.temperature,
        }
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.user_id is not None:
            payload["user"] = req.user_id
        if req.metadata:
            payload["metadata"] = req.metadata

            # Handle structured outputs if specified
        if req.structured_schema is not None:
            if isinstance(req.structured_schema, dict):
                schema = req.structured_schema
                name = schema.get(
                    "title", "structured_response"
                )                                                                         # Default name if not provided
            elif (
                BaseModel is not None
                and isinstance(req.structured_schema, type)
                and issubclass(req.structured_schema, BaseModel)
            ):
                schema_class = cast(Type[BaseModel], req.structured_schema)
                schema = schema_class.model_json_schema()                                 # Efficient schema generation
                name = schema_class.__name__                                              # Use model name for clarity
            else:
                raise ValueError(
                    "structured_schema must be a dict or Pydantic BaseModel. "
                    "Install pydantic if using models."
                )

            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": True,                                                       # Enforces strict adherence for reliability
                },
            }

        return payload

    def _filename_for(self, req: GrokRequest, resp_id: str) -> Path:
        """
        Generate a timestamped filename for JSON saving.

        Parameters
        ----------
        req : GrokRequest
            The request object.
        resp_id : str
            Response ID from API.

        Returns
        -------
        Path
            Full path to the JSON file.
        """
        ts = datetime.now(UTC).strftime(format="%Y%m%dT%H%M%SZ")
        safe_uid = (req.user_id or "no_user_id").replace("/", "_")
        return self.output_dir / f"{ts}_{safe_uid}_{resp_id}.json"                        # type: ignore[arg-type]

    async def _get_pool(self) -> Pool:
        """
        Get an existing or create a new connection pool to a Postgres database given a
        `ResolvedSettingsDict'.
        """
        db_settings: ResolvedSettingsDict = (
            self._resolved_pg_settings or responses_default_settings
        )
        if db_settings is None:
            logger.error("No DB settings available.")
            raise ValueError("DB settings required for PostgreSQL operations.")

        return await PgPoolManager.get_pool(db_settings)                                  # Efficient singleton pool

    async def _save_request_to_postgres(
        self, req: GrokRequest
    ) -> tuple[uuid.UUID, datetime]:
        """
        Insert the request into PostgreSQL and return the request_id and tstamp.

        If request_id is None, generates a new UUID and creates an immutable copy
        of GrokRequest via dataclasses.replace. This avoids mutating the original
        frozen instance while injecting the ID for persistence.

        Flow:
        1. If req.request_id is None, generate UUID and replace to new instance.
        2. Fetch or cache provider_id.
        3. Construct JSON payloads for endpoint, request, and meta.
        4. Execute INSERT query, fetching the inserted tstamp.
        5. Log success and return request_id (now non-None) with tstamp.

        Parameters
        ----------
        req : GrokRequest
            The immutable request to save; a copy is made if ID is missing.

        Returns
        -------
        tuple[uuid.UUID, datetime]
            The (generated if needed) request_id and inserted tstamp.

        Raises
        ------
        asyncpg.PostgresError
            On query execution failure (handled internally by execute_query).
        RuntimeError
            If insert succeeds but no tstamp is returned (safeguard).
        """
        if req.request_id is None:
            new_id = uuid.uuid4()                                                         # Efficient UUID generation
            req = replace(
                req, request_id=new_id
            )                                                                             # Immutable copy with ID (replace imported from `dataclasses')

        if req.request_id is None:
            err_msg = "Failed to set the request id."
            logger.info(
                msg=err_msg,
                extra={"obj": {"error": err_msg}},
            )
            raise ValueError(err_msg)

        provider_id = await self._get_provider_id()
        endpoint_json = {
            "ORG": "GROK",
            "MODEL": req.model,
            "ENDPOINT": self.base_url,
        }
        request_json = self._make_payload(req)                                            # Reuse payload as request dict
        meta_json = req.metadata or {}

        connection_pool: Pool = await self._get_pool()

        result = await execute_query(
            connection_pool,
            query_sql=(
                "INSERT INTO requests (provider_id, endpoint, request_id, request, meta) "
                "VALUES ($1, $2::jsonb, $3, $4::jsonb, $5::jsonb) "
                "RETURNING tstamp"
            ),
            params=[
                provider_id,
                endpoint_json,
                req.request_id,
                request_json,
                meta_json,
            ],
            fetch=True,
        )
        if not result:
            raise RuntimeError("Failed to insert request and retrieve tstamp.")

        inserted_tstamp: datetime = result[0]["tstamp"]
        logger.info(
            msg=f"Inserted request {req.request_id} into PostgreSQL.",
            extra={"obj": {"request_id": str(req.request_id)}},
        )
        return req.request_id, inserted_tstamp

    async def _save_response_to_postgres(
        self, resp: GrokResponse, request_id: uuid.UUID, request_tstamp: datetime
    ) -> None:
        """
        Insert the response into PostgreSQL, linking to the request via request_id and request_tstamp.

        Parameters
        ----------
        resp : GrokResponse
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
            "MODEL": resp.request.model,
            "ENDPOINT": self.base_url,
        }
        response_id_uuid = uuid.UUID(resp.response_id)                                    # Convert string to UUID
        meta_json = resp.request.metadata or {}

        connection_pool: Pool = await self._get_pool()

        await execute_query(
            connection_pool,
            query_sql=(
                "INSERT INTO responses (provider_id, endpoint, request_id, request_tstamp, "
                "response_id, response, meta) "
                "VALUES ($1, $2::jsonb, $3, $4, $5, $6::jsonb, $7::jsonb)"
            ),
            params=[
                provider_id,
                endpoint_json,
                request_id,
                request_tstamp,
                response_id_uuid,
                resp.raw,
                meta_json,
            ],
            fetch=False,
        )
        logger.info(
            msg=f"Inserted response {resp.response_id} into PostgreSQL.",
            extra={"obj": {"response_id": resp.response_id}},
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
            if self.save_mode == "postgres":
                try:
                    request_id, request_tstamp = await self._save_request_to_postgres(
                        req
                    )
                except Exception as e:
                    logger.error(
                        msg=f"Failed to save request for {req.user_id or 'unknown'}: {e}",
                        extra={"obj": {"error": str(e)}},
                    )
                    # Continue to API call even if save fails, or return e here if desired

            payload = self._make_payload(req)

            if self.conv_id:                                                              # Reuse same ID for related requests
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "x-grok-conv-id": self.conv_id,
                }
            else:
                headers = {"Authorization": f"Bearer {self.api_key}"}

            for attempt in range(self.max_retries + 1):
                try:
                    async with session.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                    ) as resp:
                        if resp.status != 200:
                            txt = await resp.text()
                            raise aiohttp.ClientError(f"HTTP {resp.status}: {txt}")

                        data = await resp.json()
                        choice = data["choices"][0]
                        grok_resp = GrokResponse(
                            request=req,
                            content=choice["message"]["content"],
                            response_id=data["id"],
                            finish_reason=choice["finish_reason"],
                            usage=data.get("usage", {}),
                            raw=data,
                        )

                        if self.save_mode == "json_files":
                            path: Path = self._filename_for(req, data["id"])
                            path.write_text(
                                data=json.dumps(data, ensure_ascii=False, indent=2)
                            )
                            logger.info(
                                msg=f"Saved response to {path.name}.",
                                extra={"obj": {"file": str(path)}},
                            )

                        elif (
                            self.save_mode == "postgres"
                            and request_id
                            and request_tstamp
                        ):
                            await self._save_response_to_postgres(
                                grok_resp, request_id, request_tstamp
                            )

                        return grok_resp

                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    json.JSONDecodeError,
                    Exception,
                ) as e:
                    if attempt == self.max_retries:
                        logger.error(
                            msg=f"Permanent failure for {req.user_id or 'unknown'}: {e}",
                            extra={"obj": {"error": str(e)}},
                        )
                        return e
                    wait = 1.5**attempt
                    logger.warning(
                        msg=f"Retry {attempt + 1}/{self.max_retries} for {req.user_id or 'unknown'}: {e}",
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
            results = await asyncio.gather(*tasks, return_exceptions=False)

        responses: list[GrokResponse] = []
        for item in results:
            if isinstance(item, Exception):
                logger.error(
                    msg=f"Task failed: {item}", extra={"obj": {"error": str(item)}}
                )
            else:
                responses.append(item)

        return responses if return_responses else None

    #  LocalWords:  infopypg's XAI infopypg conv UUID Authorization Postgres GrokRequest
    #  LocalWords:  ResolvedSettingsDict PostgresError pyrefly PgPoolManager asyncpg init
    #  LocalWords:  XAIAsyncClient bool repr asyncio postgres GrokResponse tstamp jsonb req
    #  LocalWords:  RuntimeError datetime localhost GrokMessage ValueError
