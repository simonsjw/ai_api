from pathlib import Path
from typing import Any, AsyncIterator, Literal

import ollama
from _typeshed import Incomplete
from infopypg import ResolvedSettingsDict as ResolvedSettingsDict
from openai import AsyncOpenAI

from ai_api.data_structures.grok import GrokBatchRequest as GrokBatchRequest
from ai_api.data_structures.grok import GrokBatchResponse as GrokBatchResponse
from ai_api.data_structures.grok import GrokRequest as GrokRequest
from ai_api.data_structures.grok import GrokResponse as GrokResponse
from ai_api.data_structures.grok import GrokStreamingChunk as GrokStreamingChunk
from ai_api.data_structures.grok import (
    LLMStreamingChunkProtocol as LLMStreamingChunkProtocol,
)
from ai_api.data_structures.ollama import OllamaRequest as OllamaRequest
from ai_api.data_structures.ollama import OllamaResponse as OllamaResponse
from ai_api.data_structures.ollama import OllamaStreamingChunk as OllamaStreamingChunk

class LLMClient:
    provider: Incomplete
    model: Incomplete
    settings: Incomplete
    conversation_id: Incomplete
    logger: Incomplete
    resolved_settings: ResolvedSettingsDict | None
    client: AsyncOpenAI | ollama.AsyncClient | None
    def __init__(
        self,
        provider: Literal["grok", "ollama"],
        model: str,
        settings: dict[str, Any],
        api_key: str | None = None,
        conversation_id: str | None = None,
    ) -> None: ...
    async def generate(
        self,
        request: GrokRequest | OllamaRequest,
        stream: bool = False,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> GrokResponse | OllamaResponse | AsyncIterator[LLMStreamingChunkProtocol]: ...
    async def parallel_generate(
        self,
        requests: list[GrokRequest | OllamaRequest],
        max_concurrent: int = 20,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> list[GrokResponse | OllamaResponse]: ...
    async def submit_batch(
        self, batch_request: GrokBatchRequest, persist: bool = True
    ) -> GrokBatchResponse: ...
    async def await_batch_completion(
        self, batch_id: str, poll_interval: int = 30, timeout_seconds: int | None = None
    ) -> list[GrokResponse]: ...
    async def create_ollama_model(self, modelfile: Path | str) -> dict[str, Any]: ...
    async def get_embeddings(
        self, input_text: str | list[str]
    ) -> list[float] | list[list[float]]: ...
