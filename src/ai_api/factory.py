#!/usr/bin/env python3
"""
LLM client factory.

Single public entry point for the entire `ai_api` package. Returns a
ready-to-use `BaseAsyncProviderClient` (either Grok or Ollama) with
identical method signatures. All common parameters are accepted here
and forwarded to the concrete client.

This design satisfies every requirement you listed:
- MODEL and ORG variables are the only model/host fields.
- No automatic pulling of Ollama models.
- Shared persistence, logging, concurrency, and error taxonomy.
- Foundation for later streaming, continuation_token, sys_spec, and
  resource-management modules.
"""

import asyncio
from pathlib import Path
from typing import Any, Literal

from infopypg import ResolvedSettingsDict

from .clients.base import BaseAsyncProviderClient
from .clients.grok.client import GrokConcreteClient
from .clients.ollama.client import OllamaConcreteClient
from .core.request import LLMRequest
from .data_structures import SaveMode


async def create(
    provider: Literal["grok", "ollama"],
    model: str,
    api_key: str | None = None,
    org: str | None = None,
    base_url: str | None = None,
    save_mode: SaveMode = "none",
    output_dir: Path | None = None,
    pg_settings: dict[str, Any] | ResolvedSettingsDict | None = None,
    log_location: str | dict[str, Any] | ResolvedSettingsDict = "grok_client.log",
    concurrency: int = 50,
    max_retries: int = 3,
    timeout: float = 60.0,
    set_conv_id: bool | str | None = False,
) -> BaseAsyncProviderClient:
    """
    Create and initialise a unified LLM client.

    Parameters
    ----------
    provider : Literal["grok", "ollama"]
        Which back-end to use.
    model : str
        Exact model name (the MODEL variable).
    api_key : str | None, optional
        xAI API key (required for Grok).
    org : str | None, optional
        Ollama host (the ORG variable); ignored for Grok.
    base_url : str | None, optional
        Override endpoint (advanced use).
    save_mode : SaveMode, optional
        Persistence mode.
    output_dir : Path | None, optional
        Directory for JSON files.
    pg_settings : dict[str, Any] | ResolvedSettingsDict | None, optional
        PostgreSQL settings.
    log_location : str | dict[str, Any] | ResolvedSettingsDict, optional
        Logger destination.
    concurrency : int, optional
        Maximum concurrent requests.
    max_retries : int, optional
        Retry attempts on transient failures.
    timeout : float, optional
        Request timeout (seconds).
    set_conv_id : bool | str | None, optional
        Enable conversation caching (Grok only).

    Returns
    -------
    BaseAsyncProviderClient
        Ready-to-use client (GrokConcreteClient or OllamaConcreteClient).

    Raises
    ------
    ValueError
        Invalid provider or missing required fields.
    ConnectionError
        PostgreSQL pool creation failure.
    """
    if provider == "grok":
        if not api_key:
            raise ValueError("api_key is required for provider='grok'")

        client = await GrokConcreteClient.create(
            model=model,
            api_key=api_key,
            base_url=base_url,
            save_mode=save_mode,
            output_dir=output_dir,
            pg_settings=pg_settings,
            log_location=log_location,
            concurrency=concurrency,
            max_retries=max_retries,
            timeout=timeout,
            set_conv_id=set_conv_id,
        )
        return client

    if provider == "ollama":
        # Build base_url from ORG variable (user requirement)
        host = org or "localhost"
        if ":" not in host:
            host = f"{host}:11434"
        ollama_url = base_url or f"http://{host}/v1"

        client = await OllamaConcreteClient.create(
            model=model,
            base_url=ollama_url,
            save_mode=save_mode,
            output_dir=output_dir,
            pg_settings=pg_settings,
            log_location=log_location,
            concurrency=concurrency,
            max_retries=max_retries,
            timeout=timeout,
        )
        return client

    raise ValueError(f"Unknown provider: {provider}")
