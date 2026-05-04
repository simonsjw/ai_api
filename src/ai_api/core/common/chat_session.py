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

The restored ``create_edited_branch`` functionality (via ``PersistenceManager``)
allows arbitrary history editing using committed turns from the database.

Non-chat interactions (embeddings, batch) continue to use
``PersistenceManager.persist_chat_turn`` directly.
"""

from __future__ import annotations

import uuid
from typing import Any

from ai_api.data_structures.base_objects import SaveMode
from ai_api.data_structures.ollama_objects import OllamaRequest, OllamaResponse
from ai_api.data_structures.xai_objects import xAIRequest, xAIResponse

from .errors import ClientError, wrap_error
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

    The class now fully supports the restored ad-hoc history editing via
    ``create_edited_branch``.

    Parameters
    ----------
    client : Any
        The provider client (OllamaClient or xAIClient).
    persistence_manager : PersistenceManager
        The persistence layer (must be Postgres-backed for editing features).

    Attributes
    ----------
    current_tree_id, current_branch_id, last_response_id : uuid.UUID | None
        Current conversation state.
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
        save_mode: SaveMode = SaveMode.POSTGRES,
        tree_id: uuid.UUID | None = None,
        branch_id: uuid.UUID | None = None,
        parent_response_id: uuid.UUID | None = None,
        max_depth: int | None = None,
        **generation_kwargs,
    ) -> tuple[Any, dict]:
        """Create a new conversation or continue/fork an existing branch.

        Reconstructs neutral history (if needed), calls the provider, and persists
        the result. Automatically updates session state.

        Parameters
        ----------
        new_prompt : str or list of dict
            The new user message(s).
        save_mode : SaveMode, default POSTGRES
            Persistence strategy.
        tree_id, branch_id, parent_response_id : optional
            Branching coordinates.
        max_depth : int, optional
            Limit history reconstruction depth.
        **generation_kwargs
            Passed through to the provider.

        Returns
        -------
        tuple[Any, dict]
            (provider_response, metadata)

        Raises
        ------
        ClientError
            On unsupported provider or other internal errors (wrapped).
        """
        provider = self.client.__class__.__name__.lower().replace("client", "")

        neutral_history: list[dict] = []
        if parent_response_id or tree_id or self.current_tree_id:
            try:
                neutral_history = await self.pm.reconstruct_neutral_branch(
                    tree_id=tree_id or self.current_tree_id or uuid.uuid4(),
                    branch_id=branch_id or self.current_branch_id or uuid.uuid4(),
                    up_to_response_id=parent_response_id,
                    max_depth=max_depth,
                )
            except Exception as exc:
                err = wrap_error(
                    ClientError,
                    "Failed to reconstruct history for continuation",
                    exc,
                    details={"tree_id": str(tree_id or self.current_tree_id)},
                )
                raise err from exc

        metadata = {
            "model": getattr(self.client, "model", "default"),
            **generation_kwargs,
            "save_mode": save_mode.value
            if isinstance(save_mode, SaveMode)
            else save_mode,
        }

        try:
            if provider == "ollama":
                req = OllamaRequest.from_neutral_history(
                    neutral_history, new_prompt, metadata
                )
                raw = await self.client._call_ollama(req)
                resp = OllamaResponse.from_dict(raw)
            elif provider == "xai":
                req = xAIRequest.from_neutral_history(
                    neutral_history, new_prompt, metadata
                )
                raw = await self.client._call_xai(req)
                resp = xAIResponse.from_sdk(raw)
            else:
                raise wrap_error(
                    ClientError,
                    f"Unsupported provider: {provider}",
                    None,
                    details={"provider": provider},
                )

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

        except Exception as exc:
            err = wrap_error(
                ClientError,
                "Failed during create_or_continue",
                exc,
                details={"provider": provider},
            )
            raise err from exc

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

        Uses the restored ``create_edited_branch`` functionality to apply
        arbitrary edits (remove, insert, replace turns) to committed history.

        The original branch remains completely untouched and immutable.
        After this call, ``self.current_branch_id`` points to the new branch.

        Parameters
        ----------
        edit_ops : list of dict
            Edit operations (see ``create_edited_branch`` for supported ops).
        new_branch_name : str, optional
            Human-readable name for the new branch.
        start_from_response_id, end_at_response_id, max_depth : optional
            Slice parameters for the source history.

        Returns
        -------
        dict
            Result from ``create_edited_branch`` (new_branch_id, new_response_ids, etc.).

        Raises
        ------
        ClientError
            If no active conversation or other internal error.
        """
        if not self.current_tree_id or not self.current_branch_id:
            raise wrap_error(
                ClientError,
                "Cannot edit history — no active conversation. Call create_or_continue first.",
                None,
                details={"current_tree_id": str(self.current_tree_id)},
            )

        try:
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

        except Exception as exc:
            err = wrap_error(
                ClientError,
                "Failed to edit history",
                exc,
                details={
                    "tree_id": str(self.current_tree_id),
                    "branch_id": str(self.current_branch_id),
                },
            )
            raise err from exc


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
    if "tree_id" in kwargs:
        session.current_tree_id = kwargs.pop("tree_id")
    if "branch_id" in kwargs:
        session.current_branch_id = kwargs.pop("branch_id")
    return await session.edit_history(edit_ops, **kwargs)
