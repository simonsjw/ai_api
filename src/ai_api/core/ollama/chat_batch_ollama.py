"""
Ollama Batch Chat (Simulated via Turn-Based Calls)

This module provides the dedicated implementation for batch processing of
multiple independent conversations with Ollama. Because Ollama has no native
batch endpoint (unlike xAI), batching is simulated by executing several
turn-based chats (sequentially or concurrently via ``asyncio.gather``).

The public entry point ``create_batch_chat`` is called by
``BatchOllamaClient.create_chat`` after input normalisation. This keeps the
provider architecture identical to xAI: the ``*_client.py`` file contains only
thin delegation + public API surface, while mode-specific logic lives in the
``ollama/`` sub-package.

Design Notes
------------
- Re-uses ``TurnOllamaClient`` (and therefore ``create_turn_chat_session``)
  for every item so that structured output, persistence (``save_mode``),
  branching metadata, and all Ollama-specific generation parameters are
  applied uniformly.
- ``concurrent=True`` runs all turns in parallel (useful for I/O latency);
  ``concurrent=False`` (default) runs sequentially (lower memory, easier
  debugging, preserves order strictly).
- Does **not** support per-item ``response_model`` lists (single model or
  ``None`` is applied to every conversation). This matches current Ollama
  capability; xAI batch supports heterogeneous ``response_model`` lists.
- All errors from individual turns are propagated; partial results are not
  returned on failure (fail-fast semantics).

See Also
--------
chat_turn_ollama : The turn implementation reused by every batch item.
ollama_client.BatchOllamaClient : Thin wrapper that normalises input and
    optionally unwraps single-conversation results.
xai.chat_batch_xai : Native batch implementation (different concurrency model).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional, Type

from pydantic import BaseModel

from ...data_structures.base_objects import SaveMode
from ...data_structures.ollama_objects import OllamaResponse

if TYPE_CHECKING:
    from ..ollama_client import TurnOllamaClient


async def create_batch_chat(
    client: Any,
    messages_list: list[list[dict[str, Any]]],
    model: str = "llama3.2",
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    max_tokens: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
    num_ctx: Optional[int] = None,
    stop: Optional[list[str]] = None,
    mirostat: Optional[int] = None,
    think: Optional[bool] = None,
    save_mode: SaveMode = "none",
    concurrent: bool = False,
    response_model: Type[BaseModel] | None = None,
    **kwargs: Any,
) -> list[OllamaResponse]:
    """Execute multiple chat turns against Ollama (simulated batch).

    This is the single source of truth for all Ollama batch work. It is
    invoked by ``BatchOllamaClient`` after the caller has normalised a
    possible single-conversation input into ``list[list[dict]]``.

    Each conversation in ``messages_list`` is processed by an independent
    ``TurnOllamaClient`` (re-using the full turn logic, persistence, and
    structured-output handling). The ``concurrent`` flag controls whether
    ``asyncio.gather`` is used.

    Parameters
    ----------
    client : BaseOllamaClient
        The outer client instance (provides ``logger``, ``persistence_manager``,
        ``host``, and ``timeout``). Used to construct per-item ``TurnOllamaClient``
        instances.
    messages_list : list[list[dict[str, Any]]]
        Batch of conversations. Each inner list is one complete message
        history (system/user/assistant turns). Must be non-empty for
        meaningful results.
    model : str, default "llama3.2"
        Ollama model name applied to every conversation in the batch.
    temperature : float or None, optional
        Sampling temperature (0.0 = deterministic, 1.0 = creative).
    top_p : float or None, optional
        Nucleus sampling threshold.
    top_k : int or None, optional
        Top-k sampling (limits vocabulary).
    seed : int or None, optional
        Random seed for reproducible generations.
    max_tokens : int or None, optional
        Maximum tokens to generate per response.
    repeat_penalty : float or None, optional
        Penalty for repeating tokens (Ollama-specific).
    num_ctx : int or None, optional
        Context window size in tokens.
    stop : list[str] or None, optional
        Stop sequences that halt generation.
    mirostat : int or None, optional
        Mirostat sampling mode (0=off, 1 or 2).
    think : bool or None, optional
        Enable thinking / reasoning mode on supported models.
    save_mode : {"none", "json_files", "postgres"}, default "none"
        Persistence backend applied independently to every turn.
    concurrent : bool, default False
        If True, execute all turns concurrently with ``asyncio.gather``
        (higher throughput, non-deterministic order on errors).
        If False, execute sequentially (preserves order, lower peak memory).
    response_model : type[BaseModel] or None, optional
        Pydantic model for structured output. The **same** model is used
        for every item in the batch (per-item heterogeneous models are not
        supported for Ollama; see xAI batch for that capability).
    **kwargs : Any
        Additional parameters forwarded to each ``TurnOllamaClient.create_chat``
        call (future-proofing).

    Returns
    -------
    list[OllamaResponse]
        One response object per input conversation, in the same order as
        ``messages_list``. Each response may contain a ``.parsed`` attribute
        when ``response_model`` was supplied.

    Raises
    ------
    ValueError
        If ``messages_list`` is empty.
    Exception
        Any error raised by an individual turn (e.g. HTTP failure,
        validation error, persistence error) is propagated immediately
        (fail-fast). No partial results are returned.

    Notes
    -----
    - This implementation deliberately re-creates a ``TurnOllamaClient`` for
      every batch item so that each turn has its own HTTP connection pool
      context if needed. In practice the overhead is negligible.
    - When ``concurrent=True`` the order of results is guaranteed by
      ``asyncio.gather`` (it preserves input order), but if any task fails
      the whole batch fails.
    - Persistence via ``save_mode`` happens inside each turn; the batch
      layer itself performs no additional persistence.

    Examples
    --------
    Sequential batch (default)
    >>> results = await create_batch_chat(
    ...     client,
    ...     messages_list=[conv1, conv2, conv3],
    ...     model="llama3.2",
    ...     save_mode="postgres",
    ...     concurrent=False,
    ... )
    >>> len(results)
    3
    >>> results[0].text
    'First response...'

    Concurrent batch with structured output
    >>> class Summary(BaseModel):
    ...     key_points: list[str]
    >>> results = await create_batch_chat(
    ...     client,
    ...     messages_list=[conv1, conv2],
    ...     model="llama3.2",
    ...     response_model=Summary,
    ...     concurrent=True,
    ... )
    >>> results[0].parsed.key_points
    ['point A', 'point B']

    See Also
    --------
    TurnOllamaClient.create_chat : The per-item implementation.
    BatchOllamaClient.create_chat : Input normalisation + single-result unwrap.
    """

    if not messages_list:
        raise ValueError("messages_list must contain at least one conversation")

    # Lazy import prevents circular dependency at module load time
    # (ollama_client imports chat_batch_ollama; we import TurnOllamaClient here)
    from ..ollama_client import TurnOllamaClient

    logger = getattr(client, "logger", None)
    if logger:
        logger.info(
            "Creating Ollama batch (simulated)",
            extra={
                "batch_size": len(messages_list),
                "model": model,
                "concurrent": concurrent,
            },
        )

    # Construct a TurnOllamaClient that shares the outer client's
    # logger, persistence, host and timeout. Each batch item gets its own
    # instance so HTTP connection handling remains isolated.
    turn_client = TurnOllamaClient(
        logger=client.logger,
        host=getattr(client, "host", "http://localhost:11434"),
        timeout=getattr(client, "timeout", 180),
        persistence_manager=getattr(client, "persistence_manager", None),
    )

    if concurrent:
        tasks = [
            turn_client.create_chat(
                conv,
                model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                max_tokens=max_tokens,
                repeat_penalty=repeat_penalty,
                num_ctx=num_ctx,
                stop=stop,
                mirostat=mirostat,
                think=think,
                save_mode=save_mode,
                response_model=response_model,
                **kwargs,
            )
            for conv in messages_list
        ]
        return await asyncio.gather(*tasks)
    else:
        results: list[OllamaResponse] = []
        for conv in messages_list:
            results.append(
                await turn_client.create_chat(
                    conv,
                    model,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    max_tokens=max_tokens,
                    repeat_penalty=repeat_penalty,
                    num_ctx=num_ctx,
                    stop=stop,
                    mirostat=mirostat,
                    think=think,
                    save_mode=save_mode,
                    response_model=response_model,
                    **kwargs,
                )
            )
        return results
