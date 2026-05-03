"""
Ollama turn-based (non-streaming) chat with Git-style branching support.

This module provides the coroutine ``create_turn_chat_session`` used by
``TurnOllamaClient.create_chat`` (and the batch simulator). It executes a
single HTTP POST to Ollama's ``/api/chat`` endpoint, converts the response
into an ``OllamaResponse``, and persists the interaction **exactly once**
via the unified ``PersistenceManager.persist_chat_turn`` entry point.

Branching support
-----------------
When the caller supplies ``tree_id``, ``branch_id``, ``parent_response_id``
and/or ``sequence`` via ``**kwargs``, they are forwarded to
``persist_chat_turn``. Only the **delta** (last prompt + generated response)
is stored in the ``response`` JSONB column. Full history is reconstructed
on demand by ``reconstruct_neutral_branch`` (recursive CTE).

See Also
--------
chat_stream_ollama : streaming counterpart (identical persistence path).
ai_api.core.common.chat_session.ChatSession : recommended high-level API.
ai_api.core.common.persistence.PersistenceManager.persist_chat_turn
    The single method that receives branching metadata.

Examples
--------
Basic turn with structured output and branching
>>> class Person(BaseModel):
...     name: str
...     age: int
>>> resp = await create_turn_chat_session(
...     client,
...     messages=[{"role": "user", "content": "Who is Ada Lovelace?"}],
...     model="llama3.2",
...     response_model=Person,
...     save_mode="postgres",
...     tree_id=existing_tree,
...     branch_id=new_branch,
...     parent_response_id=parent_id,
... )
>>> resp.parsed.name
'Ada Lovelace'
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from ...data_structures.ollama_objects import (
    OllamaJSONResponseSpec,
    OllamaRequest,
    OllamaResponse,
)


async def create_turn_chat_session(
    client: Any,                                                                          # BaseOllamaClient or concrete TurnOllamaClient
    messages: list[dict[str, Any]],
    model: str = "llama3.2",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    save_mode: str = "none",
    response_model: Type[BaseModel] | None = None,
    **kwargs: Any,
) -> OllamaResponse:
    """Execute a single non-streaming chat turn against Ollama and persist it.

    This is the canonical implementation for turn-based chats.  It is called by
    ``TurnOllamaClient.create_chat`` (and by the batch simulator).  Structured
    output is supported via either:

    - ``response_model`` (recommended) – a Pydantic ``BaseModel`` subclass
    - ``response_format`` passed in ``**kwargs`` – an ``OllamaJSONResponseSpec``
      or raw JSON schema dict

    The persistence step now uses the single ``persist_chat_turn`` entry point
    (the legacy ``persist_request`` + ``persist_response`` pair has been
    removed).  Branching metadata is automatically managed when the caller
    supplies ``tree_id`` / ``branch_id`` / ``parent_response_id`` via ``**kwargs``.

    Parameters
    ----------
    client : BaseOllamaClient
        The Ollama client instance (provides ``logger``, ``persistence_manager``,
        and the shared ``httpx.AsyncClient``).
    messages : list of dict
        The conversation history in OpenAI-style format
        ``[{"role": "...", "content": "..."}, ...]``.  May contain multimodal
        blocks when images are present.
    model : str, default "llama3.2"
        Ollama model name (must be pulled locally or available via the
        ``/api/tags`` endpoint).
    temperature : float, optional
        Sampling temperature (0.0 = deterministic).  If omitted the model's
        Modelfile default is used.
    max_tokens : int, optional
        Maximum number of tokens to generate.  If omitted the model's default
        applies.
    save_mode : {"none", "json_files", "postgres"}, default "none"
        Controls whether (and where) the interaction is persisted.
        When ``"postgres"`` the call goes through ``PersistenceManager``.
    response_model : type[BaseModel], optional
        Pydantic model used for structured JSON output.  When supplied it is
        automatically converted to an ``OllamaJSONResponseSpec`` and passed
        to the request.
    **kwargs
        Additional parameters forwarded to ``OllamaRequest`` (e.g.
        ``response_format``, ``tools``, ``keep_alive``, ``options``) and to
        ``persist_chat_turn`` (e.g. ``tree_id``, ``branch_id``,
        ``parent_response_id``, ``sequence``).

    Returns
    -------
    OllamaResponse
        Fully populated response object.  After persistence the caller can
        access:

        - ``response.text`` – the generated assistant message
        - ``response.parsed`` – validated Pydantic instance (if ``response_model``)
        - ``response.tool_calls`` – list of tool invocations (if any)
        - ``response.meta()`` – usage statistics, finish reason, etc.

    Raises
    ------
    httpx.HTTPStatusError
        If Ollama returns a non-2xx status (model not found, context length
        exceeded, …).
    pydantic.ValidationError
        If ``response_model`` was supplied and the generated JSON fails
        schema validation (the raw response is still returned).
    Exception
        Any error raised by ``persist_chat_turn`` is logged at WARNING level
        but does **not** abort the call – persistence failures are
        non-fatal.

    See Also
    --------
    OllamaRequest.from_neutral_history : used by higher-level branching logic
        to rebuild history before calling this function.
    persist_chat_turn : the unified persistence method (called once with the
        final response + request objects).

    Examples
    --------
    Basic turn with structured output
    >>> class Person(BaseModel):
    ...     name: str
    ...     age: int
    >>> resp = await create_turn_chat_session(
    ...     client,
    ...     messages=[{"role": "user", "content": "Who is Ada Lovelace?"}],
    ...     model="llama3.2",
    ...     response_model=Person,
    ...     save_mode="postgres",
    ... )
    >>> resp.parsed.name
    'Ada Lovelace'

    Forked conversation (branching metadata passed through **kwargs)
    >>> resp = await create_turn_chat_session(
    ...     client,
    ...     messages=history_slice,
    ...     model="llama3.2",
    ...     tree_id=existing_tree,
    ...     branch_id=new_branch,
    ...     parent_response_id=parent_id,
    ...     save_mode="postgres",
    ... )
    """
    logger = client.logger
    logger.info("Creating turn-based Ollama chat", extra={"model": model})

    # ------------------------------------------------------------------
    # 1. Structured output handling (response_model → OllamaJSONResponseSpec)
    # ------------------------------------------------------------------
    response_format: OllamaJSONResponseSpec | None = None
    if response_model is not None:
        response_format = OllamaJSONResponseSpec(model=response_model)
    elif "response_format" in kwargs:
        rf = kwargs.pop("response_format")
        if isinstance(rf, dict):
            response_format = OllamaJSONResponseSpec(model=rf)
        elif isinstance(rf, OllamaJSONResponseSpec):
            response_format = rf

    # ------------------------------------------------------------------
    # 2. Build the request object (protocol compliant)
    # ------------------------------------------------------------------
    request = OllamaRequest(
        model=model,
        input=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        save_mode=save_mode,
        response_format=response_format,
        **kwargs,
    )

    # ------------------------------------------------------------------
    # 3. Execute the HTTP call to Ollama
    # ------------------------------------------------------------------
    http_client = await client._get_http_client()
    payload = request.to_ollama_dict()
    payload["stream"] = False

    resp = await http_client.post("/api/chat", json=payload)
    resp.raise_for_status()
    raw_data = resp.json()

    response = OllamaResponse.from_dict(raw_data)

    # ------------------------------------------------------------------
    # 4. Unified persistence via persist_chat_turn (single call)
    # ------------------------------------------------------------------
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            await client.persistence_manager.persist_chat_turn(
                provider_response=response,
                provider_request=request,
                kind="chat",
                branching=True,
                # Forward branching metadata (tree/branch/parent/sequence)
                # Non-branching kwargs were already consumed by OllamaRequest.
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in {"tree_id", "branch_id", "parent_response_id", "sequence"}
                },
            )
        except Exception as exc:
            logger.warning(
                "Chat turn persistence failed (continuing)", extra={"error": str(exc)}
            )

    logger.info("Turn-based Ollama chat completed", extra={"model": model})
    return response
