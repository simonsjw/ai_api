"""
Ollama embeddings implementation (single and batch).

This module provides the canonical way to obtain embeddings from Ollama's
``/api/embeddings`` (or ``/api/embed`` in newer versions) endpoint and to
persist every embedding call through the unified ``persist_chat_turn`` path
with ``kind="embedding"`` and ``branching=False``.

Because embeddings are not conversational, they never participate in
tree/branch logic; the four branching columns are left ``NULL`` and a
``kind`` value is recorded in ``meta`` for later analytics or filtering.

Module responsibilities
-----------------------
- Accept either a single string or list of strings (batch).
- Build an ``OllamaRequest`` with the special embedding payload.
- Call the embeddings endpoint.
- Wrap the returned vector(s) in an ``OllamaEmbedResponse``.
- Persist via ``persist_chat_turn(..., kind="embedding", branching=False)``.
- Return the response object(s) so callers can access ``.embeddings``,
  ``.model``, and usage statistics.

The public ``create_embeddings`` coroutine is re-exported by
``ollama_client.py`` and also exposed as the ``embeddings`` method on
``EmbedOllamaClient``.

See Also
--------
chat_turn_ollama : chat turns use the identical persistence entry point
    (only ``kind`` and ``branching`` differ).
xai.embeddings_xai : provider-symmetric embeddings implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Union

import httpx

from ...data_structures.ollama_objects import OllamaRequest
from ..common.persistence import PersistenceManager


class OllamaEmbedResponse:
    """Lightweight container for Ollama embedding results.

    Attributes
    ----------
    model : str
        The model that produced the embedding(s).
    embeddings : list[list[float]] or list[float]
        For a single input: a 1-D list of floats.
        For batch input: a list of 1-D lists (one per input string).
    usage : dict
        Token counts and timing information returned by Ollama (when
        available).
    raw : dict
        The exact JSON payload returned by the ``/api/embeddings`` endpoint
        (useful for debugging or future fields).
    """

    def __init__(
        self,
        model: str,
        embeddings: list[Any],
        usage: dict[str, Any],
        raw: dict[str, Any],
    ):
        self.model = model
        self.embeddings = embeddings
        self.usage = usage
        self.raw = raw

    def to_neutral_format(self, branch_info: dict | None = None) -> dict[str, Any]:
        """Convert embedding result to the neutral format expected by persistence.

        The neutral turn uses ``role="embedding"`` and stores the vector(s)
        under ``content`` (JSON-serialised) so that the same ``responses``
        table can hold both chat turns and embeddings without schema changes.
        """
        return {
            "role": "embedding",
            "content": self.embeddings,
            "structured": None,
            "finish_reason": "stop",
            "usage": self.usage,
            "raw": self.raw,
            "branch_meta": branch_info or {},
        }


async def create_embeddings(
    client: Any,
    input: Union[str, list[str]],
    model: str = "nomic-embed-text",
    *,
    save_mode: str = "none",
    **kwargs: Any,
) -> OllamaEmbedResponse:
    """Generate embeddings for one or more strings and persist the call.

    This is the single source of truth for all embedding work with Ollama.
    It is called by ``EmbedOllamaClient.embeddings`` and by any code that
    needs a vector representation of text.

    Parameters
    ----------
    client : BaseOllamaClient
        Client instance providing logger, persistence_manager and HTTP client.
    input : str or list of str
        Text(s) to embed.  A single string produces a single vector; a list
        produces a list of vectors (batch mode).
    model : str, default "nomic-embed-text"
        Ollama embedding model (must be pulled locally).
    save_mode : {"none", "json_files", "postgres"}, default "none"
        When not ``"none"`` the embedding call is persisted via
        ``persist_chat_turn`` with ``kind="embedding"`` and ``branching=False``.
    **kwargs
        Extra parameters forwarded to the Ollama ``/api/embeddings`` payload
        (e.g. ``options``, ``keep_alive``) and to ``persist_chat_turn``.

    Returns
    -------
    OllamaEmbedResponse
        Container with ``.embeddings``, ``.model``, ``.usage`` and the raw
        Ollama payload.

    Raises
    ------
    httpx.HTTPStatusError
        If the embeddings endpoint returns a non-2xx status.
    Exception
        Persistence errors are logged at WARNING but do not abort the call.

    Examples
    --------
    Single embedding with Postgres persistence
    >>> emb = await create_embeddings(
    ...     client,
    ...     input="The quick brown fox",
    ...     model="nomic-embed-text",
    ...     save_mode="postgres",
    ... )
    >>> len(emb.embeddings)
    768

    Batch embeddings (no persistence)
    >>> embs = await create_embeddings(
    ...     client,
    ...     input=["doc1", "doc2", "doc3"],
    ...     model="nomic-embed-text",
    ...     save_mode="none",
    ... )
    >>> len(embs.embeddings)
    3
    """
    logger = client.logger
    logger.info(
        "Creating Ollama embeddings",
        extra={"model": model, "batch": isinstance(input, list)},
    )

    payload: dict[str, Any] = {"model": model, "input": input}
    if "options" in kwargs:
        payload["options"] = kwargs.pop("options")

    http_client = await client._get_http_client()
    resp = await http_client.post("/api/embeddings", json=payload)
    resp.raise_for_status()
    raw = resp.json()

    # Ollama returns {"embedding": [...]} for single or {"embeddings": [[...], ...]} for batch
    embeddings = raw.get("embeddings") or raw.get("embedding") or []
    usage = raw.get("usage", {})

    response = OllamaEmbedResponse(
        model=model, embeddings=embeddings, usage=usage, raw=raw
    )

    # ------------------------------------------------------------------
    # Persist via unified entry point (non-chat, no branching)
    # ------------------------------------------------------------------
    if save_mode != "none" and client.persistence_manager is not None:
        try:
            # Build a minimal request object so the prompt is recorded
            req = OllamaRequest(model=model, input=input, save_mode=save_mode, **kwargs)
            await client.persistence_manager.persist_chat_turn(
                provider_response=response,
                provider_request=req,
                kind="embedding",
                branching=False,
            )
        except Exception as exc:
            logger.warning(
                "Embedding persistence failed (continuing)", extra={"error": str(exc)}
            )

    logger.info("Ollama embeddings completed", extra={"model": model})
    return response
