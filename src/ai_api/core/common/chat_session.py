"""
High-level orchestrator for branch-aware, neutral-format chat sessions.

This module provides the developer-friendly ``ChatSession`` class and the
stateless ``create_or_continue_chat`` helper.  Both combine four steps into
a single coroutine:

1. History reconstruction via ``PersistenceManager.reconstruct_neutral_branch``
   (recursive CTE, zero duplication).
2. Provider-specific request construction via the new
   ``OllamaRequest.from_neutral_history`` / ``xAIRequest.from_neutral_history``
   methods (neutral → provider format).
3. The actual LLM call (delegated to the concrete client).
4. Persistence of the resulting turn via the unified
   ``persist_chat_turn`` (single call, correct branching metadata).

All chat interactions that need branching or forking should go through this
module.  Non-chat interactions (embeddings, batch jobs) continue to call
``persist_chat_turn`` directly with ``kind=...`` and ``branching=False``.

The module is intentionally provider-agnostic; the concrete request/response
classes are imported only to satisfy type checkers.
"""

from __future__ import annotations

import uuid
from typing import Any

from ai_api.data_structures.ollama_objects import OllamaRequest, OllamaResponse
from ai_api.data_structures.xai_objects import xAIRequest, xAIResponse

from .persistence import PersistenceManager


class ChatSession:
    """
    Stateful orchestrator for branched conversations.

    Maintains the current ``tree_id`` / ``branch_id`` / ``last_response_id``
    so callers can issue follow-up turns or forks without manually threading
    identifiers. Internally it:

    1. Calls ``PersistenceManager.reconstruct_neutral_branch`` to obtain the
       relevant history slice.
    2. Uses the provider-specific ``from_neutral_history`` class method to
       build a concrete request object.
    3. Executes the provider call (via internal client methods).
    4. Persists the result via ``persist_chat_turn``.
    5. Updates its own state for the next turn.

    Parameters
    ----------
    client : Any
        An instance of ``OllamaClient`` or ``XAIClient`` (or any future
        provider client) that exposes the low-level ``_call_*`` methods.
    persistence_manager : PersistenceManager
        The shared persistence instance (must be configured with a valid
        ``db_url`` for Postgres mode).

    Attributes
    ----------
    current_tree_id, current_branch_id, last_response_id : uuid.UUID or None
        Identifiers of the most recent turn. Updated after every successful
        ``create_or_continue`` call.
    """

    def __init__(self, client: Any, persistence_manager: PersistenceManager):
        self.client = client
        self.pm = persistence_manager
        self.current_tree_id: uuid.UUID | None = None
        self.current_branch_id: uuid.UUID | None = None
        self.last_response_id: uuid.UUID | None = None

    async def create_or_continue(
        self,
        new_prompt: str | list[dict],
        tree_id: uuid.UUID | None = None,
        branch_id: uuid.UUID | None = None,
        parent_response_id: uuid.UUID | None = None,
        max_depth: int | None = None,
        **generation_kwargs,
    ) -> tuple[Any, dict]:
        """
        Create a new conversation or continue/fork an existing branch.

        This is the primary method developers should call. It hides all the
        complexity of neutral-format conversion, recursive history reconstruction,
        and dual-write (JSONB + relational branching columns).

        Parameters
        ----------
        new_prompt : str or list of dict
            The user message (plain text or multimodal content blocks) that
            will be appended to the reconstructed history.
        tree_id : uuid.UUID, optional
            Existing tree to continue. If omitted a new tree is created.
        branch_id : uuid.UUID, optional
            Existing branch within the tree. If omitted a new branch is created.
        parent_response_id : uuid.UUID, optional
            The response this turn should be attached to. Required for any
            continuation or fork; when supplied, history is reconstructed
            starting from that point.
        max_depth : int, optional
            Maximum number of prior turns to include when reconstructing
            history (safety / performance knob for very long conversations).
        **generation_kwargs
            Any additional generation parameters (temperature, max_tokens,
            response_format, tools, etc.) that will be passed through to the
            provider request constructor.

        Returns
        -------
        tuple
            ``(response_object, saved_meta)`` where ``response_object`` is the
            concrete provider response (``OllamaResponse`` or ``xAIResponse``)
            and ``saved_meta`` is the dict returned by ``persist_chat_turn``
            containing the new ``tree_id``, ``branch_id``, ``sequence``, etc.

        See Also
        --------
        create_or_continue_chat : stateless convenience wrapper.
        PersistenceManager.persist_chat_turn : the underlying persistence call.
        """
        provider = self.client.__class__.__name__.lower().replace("client", "")

        # 1. Reconstruct neutral history from Postgres (only relevant slice)
        neutral_history: list[dict] = []
        if parent_response_id or tree_id:
            neutral_history = await self.pm.reconstruct_neutral_branch(
                tree_id=tree_id or self.current_tree_id or uuid.uuid4(),
                branch_id=branch_id or self.current_branch_id or uuid.uuid4(),
                up_to_response_id=parent_response_id,                                     # stop before including the parent we're forking from
                max_depth=max_depth,
            )

        # 2. Build provider-specific request using new neutral method
        metadata = {
            "model": getattr(self.client, "model", "default"),
            **generation_kwargs,
            "save_mode": "postgres",
        }

        if provider == "ollama":
            req = OllamaRequest.from_neutral_history(
                neutral_history, new_prompt, metadata
            )
            raw = await self.client._call_ollama(req)                                     # assume internal method
            resp = OllamaResponse.from_dict(raw)
        elif provider == "xai":
            req = xAIRequest.from_neutral_history(neutral_history, new_prompt, metadata)
            raw = await self.client._call_xai(req)
            resp = xAIResponse.from_sdk(raw)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # 3. Persist with full branch context (delta only — no history duplication)
        saved = await self.pm.persist_chat_turn(
            provider_response=resp,
            provider_request=req,
            tree_id=tree_id or self.current_tree_id,
            branch_id=branch_id or self.current_branch_id,
            parent_response_id=parent_response_id or self.last_response_id,
            neutral_history_slice=neutral_history,
        )

        # Update session state
        self.current_tree_id = saved["tree_id"]
        self.current_branch_id = saved["branch_id"]
        self.last_response_id = saved.get("parent_response_id")                           # or new response_id

        return resp, saved


# Standalone helper if you prefer not to use the class
async def create_or_continue_chat(
    client: Any, pm: PersistenceManager, new_prompt: str | list[dict], **kwargs
) -> tuple[Any, dict]:
    """
    Stateless convenience wrapper around ``ChatSession``.

    Creates a temporary ``ChatSession``, calls ``create_or_continue``, and
    returns the result. Useful for one-off calls or when you do not need
    to maintain session state between turns.

    Parameters
    ----------
    client : Any
        Provider client instance (Ollama or xAI).
    pm : PersistenceManager
        Configured persistence manager.
    new_prompt : str or list of dict
        The user prompt for this turn.
    **kwargs
        Forwarded to ``ChatSession.create_or_continue`` (tree_id, branch_id,
        parent_response_id, max_depth, generation parameters, etc.).

    Returns
    -------
    tuple
        Same as ``ChatSession.create_or_continue``.
    """
    session = ChatSession(client, pm)
    return await session.create_or_continue(new_prompt, **kwargs)
