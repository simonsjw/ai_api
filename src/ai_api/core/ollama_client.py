"""High-level asynchronous client for Ollama (local) using native API.

Mirrors the exact public API and behaviour of xai_client.py so you can
swap providers with minimal code changes:

    client = OllamaClient(logger=logger, host="http://localhost:11434", ...)
    response = await client.create_chat(messages=..., model="llama3.2", ...)

Fully supports:
- Turn-based (non-streaming)
- Streaming (with real-time persistence)
- Structured JSON output via OllamaJSONResponseSpec
- Multimodal (base64 images in messages)
- Batching (simulated via concurrent turn calls — Ollama has no native batch endpoint)
- Embeddings via EmbedOllamaClient
- Your existing PersistenceManager (reused unchanged)
- Same SaveMode, logging pattern, and error wrapping style
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal, Optional, Type, Union, cast

import httpx
from pydantic import BaseModel

from ..data_structures.ollama_objects import (
    LLMStreamingChunkProtocol,
    OllamaInput,
    OllamaJSONResponseSpec,
    OllamaMessage,
    OllamaRequest,
    OllamaResponse,
    OllamaRole,
    OllamaStreamingChunk,
    SaveMode,
)
from .common.persistence import PersistenceManager
from .ollama.chat_stream_ollama import generate_stream_and_persist
from .ollama.chat_turn_ollama import create_turn_chat_session
from .ollama.embeddings_ollama import OllamaEmbedResponse, create_embeddings

ChatMode = Literal["turn", "stream", "batch"]


class BaseOllamaClient:
    """Shared base with HTTP client lifecycle."""

    def __init__(
        self,
        logger: logging.Logger,
        host: str = "http://localhost:11434",
        timeout: Optional[int] = 180,                                                     # Ollama can be slower on large models
        persistence_manager: "PersistenceManager | None" = None,
        **kwargs: Any,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.logger = logger
        self.persistence_manager = persistence_manager
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.host,
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._http_client

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def get_model_options(self, model: str) -> dict[str, Any]:
        """Fetch the model's default generation parameters from Ollama (/api/show).

        This returns the parameters defined in the Modelfile (temperature, top_k,
        num_ctx, etc.). Very useful for understanding what defaults a model ships with
        before overriding them.
        """
        http_client = await self._get_http_client()
        try:
            resp = await http_client.post("/api/show", json={"model": model})
            resp.raise_for_status()
            data = resp.json()
            # Ollama returns 'parameters' as a string or dict depending on version
            params = data.get("parameters") or {}
            if isinstance(params, str):
                # Parse the Modelfile-style string into a dict (simple parser)
                parsed: dict[str, Any] = {}
                for line in params.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if " " in line:
                            key, value = line.split(" ", 1)
                            parsed[key.lower()] = value.strip()
                return parsed
            return params
        except Exception as exc:
            self.logger.warning(
                f"Failed to fetch model options for {model}", extra={"error": str(exc)}
            )
            return {}


class TurnOllamaClient(BaseOllamaClient):
    async def create_chat(
        self,
        messages: list[dict] | None = None,
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
        response_model: type["BaseModel"] | None = None,
        **kwargs: Any,
    ) -> OllamaResponse:
        return await create_turn_chat_session(
            self,
            messages or [],
            model=model,
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


class StreamOllamaClient(BaseOllamaClient):
    """Streaming client for Ollama (now thin and fully delegated)."""

    async def create_chat(
        self,
        messages: list[dict],
        model: str,
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
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamingChunkProtocol]:
        """Streaming Ollama chat – delegates persistence + streaming to generate_stream_and_persist."""
        self.logger.info(
            "Starting Ollama streaming chat",
            extra={"model": model, "save_mode": save_mode},
        )

        ollama_input = OllamaInput.from_list(messages)
        request = OllamaRequest(
            model=model,
            input=ollama_input,
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
            **kwargs,
        )

        http_client = await self._get_http_client()

        async for chunk in generate_stream_and_persist(
            self.logger,
            self.persistence_manager,
            http_client,
            request,
            save_mode=save_mode,
        ):
            yield chunk


class BatchOllamaClient(BaseOllamaClient):
    """Batch support for Ollama (safe by default).

    Ollama has no native batch endpoint. We simulate batching by running
    multiple independent turn-based chats.

    - `concurrent=False` (default): Sequential execution — safest when your
      model barely fits in GPU memory.
    - `concurrent=True`: Uses asyncio.gather for parallel requests (only use
      if you have headroom on GPU or run multiple Ollama instances).

    GPU memory is logged before/after each batch (best-effort via pynvml or torch).
    A warning is emitted if free VRAM looks critically low.

    Accepts either a single conversation (list[dict]) or list of conversations.
    Returns OllamaResponse or list[ OllamaResponse ] accordingly.
    """

    async def create_chat(
        self,
        messages: list[dict] | list[list[dict]],
        model: str,
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
        response_model: type["BaseModel"] | None = None,
        concurrent: bool = False,
        **kwargs: Any,
    ) -> Union[OllamaResponse, list[OllamaResponse]]:
        """Run one or more chat conversations (sequentially by default for safety)."""
        n_conversations = (
            1 if (messages and isinstance(messages[0], dict)) else len(messages)
        )
        self.logger.info(
            "Ollama batch chat started",
            extra={
                "model": model,
                "n_conversations": n_conversations,
                "concurrent": concurrent,
            },
        )

        # Normalize to list of conversations (type-safe for Pyrefly)
        if messages and isinstance(messages[0], dict):
            conversations: list[list[dict]] = [cast(list[dict], messages)]
            is_single = True
        else:
            conversations = cast(list[list[dict]], messages)
            is_single = False

            # GPU memory check (best effort, no hard dependency)
        self._log_gpu_memory("before batch", model)

        # Create a fresh turn client (reuses same config)
        turn_client = TurnOllamaClient(
            logger=self.logger,
            host=self.host,
            timeout=self.timeout,
            persistence_manager=self.persistence_manager,
        )

        # Build tasks
        tasks = [
            turn_client.create_chat(
                messages=convo,
                model=model,
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
            for convo in conversations
        ]

        # Execute (safe sequential by default)
        if concurrent:
            self.logger.warning(
                "Concurrent batch requested — ensure sufficient GPU memory headroom",
                extra={"model": model, "n_conversations": len(conversations)},
            )
            results = await asyncio.gather(*tasks, return_exceptions=False)
        else:
            results: list[OllamaResponse] = []
            for i, task in enumerate(tasks):
                self.logger.debug(f"Running batch item {i + 1}/{len(tasks)}")
                results.append(await task)

        self._log_gpu_memory("after batch", model)

        self.logger.info(
            "Ollama batch completed",
            extra={"model": model, "n_results": len(results), "concurrent": concurrent},
        )
        return results[0] if is_single else results

    def _log_gpu_memory(self, stage: str, model: str) -> None:
        """Best-effort GPU memory logging. Graceful if pynvml/torch unavailable."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = mem_info.used / (1024**3)
            total_gb = mem_info.total / (1024**3)
            free_gb = mem_info.free / (1024**3)

            self.logger.info(
                f"GPU memory {stage}",
                extra={
                    "model": model,
                    "used_gb": round(used_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "free_gb": round(free_gb, 2),
                },
            )

            if (
                free_gb < 1.5
            ):                                                                            # Heuristic: less than ~1.5 GB free is risky for most models
                self.logger.warning(
                    "Low GPU memory detected — concurrent batching may cause OOM or slow unloading",
                    extra={"free_gb": round(free_gb, 2), "model": model},
                )
        except ImportError:
            # Try torch as fallback
            try:
                import torch

                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    self.logger.info(
                        f"GPU memory {stage} (torch)",
                        extra={
                            "model": model,
                            "allocated_gb": round(allocated, 2),
                            "reserved_gb": round(reserved, 2),
                        },
                    )
            except Exception:
                self.logger.debug(
                    "GPU memory monitoring unavailable (no pynvml or torch.cuda)"
                )
        except Exception as exc:
            self.logger.debug(f"GPU memory query failed: {exc}")


def OllamaClient(
    logger: logging.Logger,
    host: str = "http://localhost:11434",
    *,
    mode: ChatMode = "turn",
    timeout: Optional[int] = 180,
    persistence_manager: "PersistenceManager | None" = None,
    **kwargs: Any,
) -> BaseOllamaClient:
    """Factory – exactly mirrors XAIClient factory."""
    client_map: dict[ChatMode, Type[BaseOllamaClient]] = {
        "turn": TurnOllamaClient,
        "stream": StreamOllamaClient,
        "batch": BatchOllamaClient,
    }

    ClientClass = client_map.get(mode)
    if ClientClass is None:
        raise ValueError(
            f"Unsupported mode '{mode}'. Must be one of: {list(client_map.keys())}"
        )

    return ClientClass(
        logger=logger,
        host=host,
        timeout=timeout,
        persistence_manager=persistence_manager,
        **kwargs,
    )


class EmbedOllamaClient(BaseOllamaClient):
    async def create_embeddings(self, *args, **kwargs) -> "OllamaEmbedResponse":
        from .ollama.embeddings_ollama import EmbedOllamaClient as EmbedImpl
        from .ollama.embeddings_ollama import create_embeddings

        impl = EmbedImpl(
            logger=self.logger,
            host=self.host,
            timeout=self.timeout,
            persistence_manager=self.persistence_manager,
        )
        return await create_embeddings(impl, *args, **kwargs)
