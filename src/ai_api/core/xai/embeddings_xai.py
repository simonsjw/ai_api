"""
xAI Embeddings (Non-Conversational Path)

This module provides the dedicated implementation for generating text
embeddings against the xAI API. It is the single source of truth called by
``EmbedXAIClient.create_chat`` (which delegates here for symmetry with
Ollama's ``embeddings_ollama.py``).

The implementation uses xAI's OpenAI-compatible ``/v1/embeddings`` endpoint
(``text-embedding-3-large`` and similar models). It supports both single-string
and batch (list-of-strings) input, returns a rich ``XAIEmbedResponse`` object,
and optionally persists the call via the unified ``persist_chat_turn`` entry
point with ``kind="embedding"`` and ``branching=False`` (embeddings are never
part of a Git-style conversation tree).

Error Handling (High Standard)
------------------------------
All network / API failures are wrapped using ``wrap_xai_api_error`` (from
``errors_xai.py``) which produces a properly typed ``xAIAPIError`` with
``__cause__`` and ``details``. HTTP status errors are specially caught and
enriched with status code. Persistence errors are logged at WARNING level
but never abort the embedding call (graceful degradation). This matches the
rigorous approach used throughout the xAI provider modules and the shared
``core/common/errors.py`` hierarchy.

Design Alignment
----------------
- Mirrors ``embeddings_ollama.py`` structure and ``OllamaEmbedResponse``.
- Uses ``xAIRequest`` (from ``data_structures.xai_objects``) for the persisted
  request side so that the neutral schema remains consistent.
- ``XAIEmbedResponse`` implements ``to_neutral_format`` so the same
  ``responses`` table / recursive CTE works for embeddings without schema
  changes.
- No branching metadata is accepted or forwarded (embeddings are orthogonal
  to conversation history).

See Also
--------
ai_api.core.xai.xai_client.EmbedXAIClient
ai_api.core.ollama.embeddings_ollama
ai_api.core.common.persistence.PersistenceManager.persist_chat_turn
ai_api.core.xai.errors_xai (wrap_xai_api_error, xAIAPIError)
"""

from __future__ import annotations

import logging
from typing import Any, Union

import httpx

from ...data_structures.xai_objects import xAIRequest
from .errors_xai import wrap_xai_api_error


class XAIEmbedResponse:
    """Container for xAI embedding results (OpenAI-compatible response format).

    This class normalises the raw JSON returned by ``/v1/embeddings`` into a
    convenient Python object while preserving the full payload for debugging
    or future-proofing.

    Attributes
    ----------
    model : str
        The embedding model that produced the vectors (e.g. "text-embedding-3-large").
    embeddings : list[list[float]] or list[float]
        - Single input: a 1-D list of floats (the embedding vector).
        - Batch input: a list of 1-D lists, one vector per input string.
    usage : dict
        Token usage statistics returned by the API (prompt_tokens, total_tokens, etc.).
        May be empty for some models.
    raw : dict
        The exact JSON payload returned by the xAI embeddings endpoint. Useful
        for inspecting additional fields or for advanced use-cases.

    Methods
    -------
    to_neutral_format(branch_info=None)
        Returns a dict in the neutral turn format expected by
        ``PersistenceManager.persist_chat_turn`` (role="embedding").

    Examples
    --------
    >>> resp = XAIEmbedResponse(
    ...     model="text-embedding-3-large",
    ...     embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    ...     usage={"prompt_tokens": 10, "total_tokens": 10},
    ...     raw={"object": "list", "data": [...]},
    ... )
    >>> len(resp.embeddings)
    2
    """

    def __init__(
        self,
        model: str,
        embeddings: list[Any],
        usage: dict[str, Any],
        raw: dict[str, Any],
    ) -> None:
        self.model = model
        self.embeddings = embeddings
        self.usage = usage
        self.raw = raw

    def to_neutral_format(
        self, branch_info: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Convert embedding result to the neutral format used by persistence.

        The neutral turn stores embeddings under ``content`` (as JSON-serialisable
        list(s) of floats) with ``role="embedding"``. This allows the same
        ``responses`` table and Git-branching infrastructure to hold both chat
        turns and embedding calls without any schema modification.

        Parameters
        ----------
        branch_info : dict or None, optional
            Optional branching metadata (ignored for embeddings; always empty).

        Returns
        -------
        dict
            Neutral representation ready for ``persist_chat_turn``.
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
    model: str = "text-embedding-3-large",
    *,
    save_mode: str = "none",
    **kwargs: Any,
) -> XAIEmbedResponse:
    """Generate one or more text embeddings using the xAI embeddings endpoint.

    This is the canonical implementation for all xAI embedding work. It is
    invoked by ``EmbedXAIClient.create_chat`` and may be called directly by
    advanced users who need low-level control.

    The call is made against xAI's OpenAI-compatible ``/v1/embeddings`` route.
    Both single-string and batch (list) inputs are supported. Extra keyword
    arguments (e.g. ``dimensions``, ``encoding_format``) are forwarded
    verbatim to the API payload.

    Persistence
    -----------
    When ``save_mode != "none"`` the embedding call is recorded via the
    unified ``persist_chat_turn`` helper with ``kind="embedding"`` and
    ``branching=False``. This keeps the conversation tree clean (embeddings
    never participate in Git-style rebase/edit_history workflows).

    Parameters
    ----------
    client : BaseXAIClient (or subclass)
        The client instance that supplies ``logger``, ``persistence_manager``,
        and the shared ``_get_http_client()`` coroutine.
    input : str or list of str
        Text(s) to embed.
        - str → single embedding vector returned.
        - list[str] → list of embedding vectors (batch mode).
    model : str, default "text-embedding-3-large"
        xAI embedding model identifier. Must be a model that supports the
        embeddings endpoint (currently OpenAI-compatible names are accepted).
    save_mode : {"none", "json_files", "postgres"}, default "none"
        Persistence backend. When not ``"none"`` the interaction is written
        to the chosen store using ``kind="embedding"``.
    **kwargs : Any
        Additional fields passed directly into the embeddings request payload
        (e.g. ``dimensions=1024``, ``encoding_format="float"``) and also
        forwarded to ``xAIRequest`` for persistence.

    Returns
    -------
    XAIEmbedResponse
        Rich response object containing the embedding vector(s), usage stats,
        and the original raw payload.

    Raises
    ------
    xAIAPIError
        Any failure at the HTTP / API level is wrapped by
        ``wrap_xai_api_error`` (from ``errors_xai.py``). This includes
        network errors, authentication failures, rate limits, and non-2xx
        HTTP responses. The original exception is attached as ``__cause__``.
    httpx.HTTPStatusError
        Re-raised after wrapping when the embeddings endpoint returns a
        non-success status code.
    Exception
        Persistence errors are caught, logged at WARNING level, and do not
        propagate (the embedding result is still returned).

    Examples
    --------
    Single embedding (no persistence)
    >>> emb = await create_embeddings(
    ...     client,
    ...     input="The quick brown fox jumps over the lazy dog.",
    ...     model="text-embedding-3-large",
    ...     save_mode="none",
    ... )
    >>> len(emb.embeddings)
    3072
    >>> emb.model
    'text-embedding-3-large'

    Batch embeddings with Postgres persistence
    >>> embs = await create_embeddings(
    ...     client,
    ...     input=["doc1 text", "doc2 text", "doc3 text"],
    ...     model="text-embedding-3-large",
    ...     save_mode="postgres",
    ...     dimensions=1024,
    ... )
    >>> len(embs.embeddings)
    3
    >>> len(embs.embeddings[0])
    1024

    Notes
    -----
    - xAI currently exposes embeddings via an OpenAI-compatible interface.
      Model names such as ``text-embedding-3-large`` are therefore valid.
    - Embeddings are intentionally excluded from conversation branching.
      The ``branching=False`` flag ensures they do not appear in
      ``reconstruct_neutral_branch`` results.
    - The ``XAIEmbedResponse`` object satisfies the structural requirements
      expected by ``persist_chat_turn`` (it exposes ``to_neutral_format``).

    See Also
    --------
    XAIEmbedResponse.to_neutral_format
    ai_api.core.xai.errors_xai.wrap_xai_api_error
    ai_api.core.common.persistence.PersistenceManager.persist_chat_turn
    """

    logger = client.logger
    logger.info(
        "Creating xAI embeddings",
        extra={"model": model, "batch": isinstance(input, list)},
    )

    # Build payload – forward any extra kwargs (dimensions, encoding_format, ...)
    payload: dict[str, Any] = {"model": model, "input": input}
    # Merge user-supplied extra fields (they take precedence)
    payload.update({k: v for k, v in kwargs.items() if k not in ("save_mode",)})

    http_client = await client._get_http_client()

    # ------------------------------------------------------------------
    # API call with high-standard error trapping
    # ------------------------------------------------------------------
    try:
        resp = await http_client.post("/v1/embeddings", json=payload)
        resp.raise_for_status()
        raw: dict[str, Any] = resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        logger.error(
            "xAI embeddings endpoint returned non-2xx status",
            extra={"status_code": status, "url": str(exc.request.url)},
        )
        raise wrap_xai_api_error(
            exc, f"xAI embeddings failed with HTTP {status}"
        ) from exc
    except Exception as exc:
        logger.error(
            "xAI embeddings request failed",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise wrap_xai_api_error(exc, "xAI embeddings request failed") from exc

    # ------------------------------------------------------------------
    # Normalise response into XAIEmbedResponse
    # ------------------------------------------------------------------
    data = raw.get("data", [])
    if isinstance(input, str) and len(data) == 1:
        embeddings: list[Any] = data[0].get("embedding", [])
    else:
        embeddings = [item.get("embedding", []) for item in data]

    usage: dict[str, Any] = raw.get("usage", {})

    response = XAIEmbedResponse(
        model=model,
        embeddings=embeddings,
        usage=usage,
        raw=raw,
    )

    # ------------------------------------------------------------------
    # Optional persistence (non-chat, no branching)
    # ------------------------------------------------------------------
    if save_mode != "none" and getattr(client, "persistence_manager", None) is not None:
        try:
            # Construct a proper xAIRequest so we stay inside the protocol
            embed_request = xAIRequest(
                model=model,
                input=" | ".join(input) if isinstance(input, list) else input,
                save_mode=save_mode,
                **{k: v for k, v in kwargs.items() if k not in ("save_mode",)},
            )
            await client.persistence_manager.persist_chat_turn(
                provider_response=response,
                provider_request=embed_request,
                kind="embedding",
                branching=False,
            )
        except Exception as exc:
            logger.warning(
                "Embedding persistence failed (continuing gracefully)",
                extra={"error": str(exc)},
            )

    logger.info("xAI embeddings completed successfully", extra={"model": model})
    return response
