"""
High-level orchestrator for branch-aware, neutral-format chat sessions.

This module provides the developer-friendly ``ChatSession`` class and the
stateless ``create_or_continue_chat`` / ``edit_chat_history`` helpers.

All chat interactions that need branching, forking, or history editing should
go through this module. The design follows a Git-like model:

- ``create_or_continue`` = normal commit / continue on current branch
- ``edit_history``     = rebase: create a new branch with edited history,
                         then switch the session to it (original branch is
                         left completely untouched and immutable).

Non-chat interactions (embeddings, batch) continue to use
``PersistenceManager.persist_chat_turn`` directly.
"""

from __future__ import annotations

import uuid
from typing import Any

from ai_api.data_structures.ollama_objects import OllamaRequest, OllamaResponse
from ai_api.data_structures.xai_objects import xAIRequest, xAIResponse

from .persistence import PersistenceManager

__all__: list[str] = ["ChatSession", "create_or_continue_chat", "edit_chat_history"]


class ChatSession:
    """
    Stateful orchestrator for branched conversations (Git-style).

    Maintains ``current_tree_id`` / ``current_branch_id`` / ``last_response_id``
    so callers can issue follow-ups, forks, or history edits without manually
    threading identifiers.

    After calling ``edit_history(...)``, the session automatically switches to
    the newly created branch (exactly like `git checkout` after a rebase).
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
        """Create a new conversation or continue/fork an existing branch."""
        provider = self.client.__class__.__name__.lower().replace("client", "")

        neutral_history: list[dict] = []
        if parent_response_id or tree_id or self.current_tree_id:
            neutral_history = await self.pm.reconstruct_neutral_branch(
                tree_id=tree_id or self.current_tree_id or uuid.uuid4(),
                branch_id=branch_id or self.current_branch_id or uuid.uuid4(),
                up_to_response_id=parent_response_id,
                max_depth=max_depth,
            )

        metadata = {
            "model": getattr(self.client, "model", "default"),
            **generation_kwargs,
            "save_mode": "postgres",
        }

        if provider == "ollama":
            req = OllamaRequest.from_neutral_history(
                neutral_history, new_prompt, metadata
            )
            raw = await self.client._call_ollama(req)
            resp = OllamaResponse.from_dict(raw)
        elif provider == "xai":
            req = xAIRequest.from_neutral_history(neutral_history, new_prompt, metadata)
            raw = await self.client._call_xai(req)
            resp = xAIResponse.from_sdk(raw)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        saved = await self.pm.persist_chat_turn(
            provider_response=resp,
            provider_request=req,
            tree_id=tree_id or self.current_tree_id,
            branch_id=branch_id or self.current_branch_id,
            parent_response_id=parent_response_id or self.last_response_id,
            neutral_history_slice=neutral_history,
        )

        self.current_tree_id = saved["tree_id"]
        self.current_branch_id = saved["branch_id"]
        self.last_response_id = saved.get("parent_response_id") or saved.get(
            "response_id"
        )

        return resp, saved

    async def edit_history(
        self,
        edit_ops: list[dict[str, Any]],
        new_branch_name: str | None = None,
        start_from_response_id: uuid.UUID | None = None,
        end_at_response_id: uuid.UUID | None = None,
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        """
        Create a new edited branch (Git rebase) and switch the session to it.

        This is the primary API for arbitrary history editing:
        - Remove irrelevant early turns
        - Insert clarifying prompts
        - Replace turns with corrected versions
        - etc.

        The original branch remains completely untouched and immutable.
        After this call, ``self.current_branch_id`` points to the new branch,
        so subsequent ``create_or_continue`` calls continue from the edited history.

        Parameters
        ----------
        edit_ops : list of dict
            Edit operations (see ``PersistenceManager.create_edited_branch`` for
            supported ops: ``remove_turns``, ``insert_turn_after``, ``replace_turn``).
        new_branch_name : str, optional
            Human-readable name stored in ``Conversations.branch_metadata``.
        start_from_response_id, end_at_response_id, max_depth
            Optional slice of the current branch to edit (defaults to whole branch).

        Returns
        -------
        dict
            Result from ``create_edited_branch`` (new_branch_id, new_response_ids,
            edited_history, operations_applied, etc.).
        """
        if not self.current_tree_id or not self.current_branch_id:
            raise RuntimeError(
                "Cannot edit history — no active conversation. "
                "Call create_or_continue first or pass tree_id/branch_id."
            )

        result = await self.pm.create_edited_branch(
            tree_id=self.current_tree_id,
            source_branch_id=self.current_branch_id,
            edit_ops=edit_ops,
            new_branch_name=new_branch_name,
            start_from_response_id=start_from_response_id,
            end_at_response_id=end_at_response_id,
            max_depth=max_depth,
        )

        # Switch session state to the new branch (like `git checkout` after rebase)
        self.current_branch_id = result["new_branch_id"]
        if result.get("new_response_ids"):
            self.last_response_id = result["new_response_ids"][-1]

        return result


# ====================== Stateless convenience wrappers ======================


async def create_or_continue_chat(
    client: Any, pm: PersistenceManager, new_prompt: str | list[dict], **kwargs
) -> tuple[Any, dict]:
    """Stateless wrapper around ``ChatSession.create_or_continue``."""
    session = ChatSession(client, pm)
    return await session.create_or_continue(new_prompt, **kwargs)


async def edit_chat_history(
    client: Any, pm: PersistenceManager, edit_ops: list[dict], **kwargs
) -> dict[str, Any]:
    """
    Stateless wrapper around ``ChatSession.edit_history``.

    Creates a temporary session, calls edit, and returns the result.
    Useful when you don't need to maintain state across multiple operations.
    """
    session = ChatSession(client, pm)
    # If caller passed tree_id/branch_id, set them on the temp session first
    if "tree_id" in kwargs:
        session.current_tree_id = kwargs.pop("tree_id")
    if "branch_id" in kwargs:
        session.current_branch_id = kwargs.pop("branch_id")
    return await session.edit_history(edit_ops, **kwargs)
