#!/usr/bin/env python3
"""
Abstract base class for all LLM providers.

This module defines `BaseAsyncProviderClient`, the contract that every
concrete client (Grok, Ollama, future providers) must satisfy. It
guarantees a uniform public API while allowing each provider to
implement its own HTTP payload mapping, error taxonomy, and streaming
format.

Design priorities (in order):
1. Efficiency — abstract methods only; no runtime overhead.
2. Clarity — every method documents exactly what a provider must do.
3. Readability — short methods, full NumPy-style docstrings, inline
   comments after column 90.

All persistence, logging, and concurrency logic is delegated to the
concrete implementations so that shared behaviour (JSON files,
PostgreSQL partitioned tables, structured logger) is reused without
duplication.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar, Literal

from infopypg import ResolvedSettingsDict

from ..core.request import LLMRequest
from ..core.response import LLMResponse, StreamingChunk
from ..data_structures import SaveMode

type ProviderLiteral = Literal["grok", "ollama"]


class BaseAsyncProviderClient(ABC):
    """
    Abstract base for all LLM providers.

    Concrete subclasses must implement every abstract method. The factory
    guarantees that the same `submit_batch` and `stream` signatures are
    presented to the caller regardless of provider.

    Parameters (set by factory)
    ---------------------------
    provider_name : ProviderLiteral
        Class attribute identifying the back-end.
    model : str
        The model passed to the factory.
    """

    provider_name: ClassVar[Literal["grok", "ollama"]]

    @abstractmethod
    async def submit_batch(
        self,
        requests: list[LLMRequest],
        return_responses: bool = True,
    ) -> list[LLMResponse] | None:
        """
        Submit a batch of requests in parallel.

        Parameters
        ----------
        requests : list[LLMRequest]
            Unified requests (already validated).
        return_responses : bool, optional
            Whether to return the list of responses (default True).

        Returns
        -------
        list[LLMResponse] | None
            Successful responses or None (when `return_responses=False`).

        Flow
        ----
        1. Acquire semaphore.
        2. Map each LLMRequest → provider payload.
        3. Execute HTTP calls with retries.
        4. Convert raw responses → LLMResponse.
        5. Persist (JSON or PostgreSQL) and log.
        """
        ...

    @abstractmethod
    def stream(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[StreamingChunk]:
        """
        Stream tokens for a single request.

        Yields
        ------
        StreamingChunk
            One chunk per token (or tool-call delta).

        Raises
        ------
        NotImplementedError
            If the provider or model does not support streaming.
        """
        ...

    @abstractmethod
    def can_stream(self) -> bool:
        """
        Quick capability check.

        Returns
        -------
        bool
            True if this provider/model supports streaming.
        """
        ...

    @abstractmethod
    def required_vram_gb(self, request: LLMRequest) -> float | None:
        """
        Estimate VRAM required for this request (Ollama only).

        Returns
        -------
        float | None
            Estimated VRAM in GB; None for remote providers.
        """
        ...

    async def _persist_streamed_response(
        self, request: LLMRequest, resp: LLMResponse
    ) -> None:
        """
        Persist a streamed response (optional hook).

        Concrete clients (GrokConcreteClient and OllamaConcreteClient)
        should override this method to reuse their existing
        `_save_request_to_postgres` and `_save_response_to_postgres`
        logic.

        The default is a safe no-op.
        """
        ...
