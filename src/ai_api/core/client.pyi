import ollama
from _typeshed import Incomplete
from ai_api.data_structures.grok import GrokBatchRequest, GrokBatchResponse, GrokRequest, GrokResponse, LLMStreamingChunkProtocol
from ai_api.data_structures.ollama import OllamaRequest, OllamaResponse
from infopypg import ResolvedSettingsDict as ResolvedSettingsDict
from logging import Logger
from openai import AsyncOpenAI
from pathlib import Path
from typing import Any, AsyncIterator, Literal

class LLMClient:
    provider: Incomplete
    model: Incomplete
    settings: Incomplete
    conversation_id: Incomplete
    logger: Logger | None
    resolved_settings: ResolvedSettingsDict | None
    client: AsyncOpenAI | ollama.AsyncClient | None
    def __init__(self, provider: Literal['grok', 'ollama'], model: str, settings: dict[str, Any], api_key: str | None = None, conversation_id: str | None = None, logger: Logger | None = None) -> None: ...
    async def generate(self, request: GrokRequest | OllamaRequest, stream: bool = False, use_cache: bool = True, **kwargs: Any) -> GrokResponse | OllamaResponse | AsyncIterator[LLMStreamingChunkProtocol]: ...
    async def parallel_generate(self, requests: list[GrokRequest | OllamaRequest], max_concurrent: int = 20, use_cache: bool = True, **kwargs: Any) -> list[GrokResponse | OllamaResponse]: ...
    async def submit_batch(self, batch_request: GrokBatchRequest, persist: bool = True) -> GrokBatchResponse: ...
    async def await_batch_completion(self, batch_id: str, poll_interval: int = 30, timeout_seconds: int | None = None) -> list[GrokResponse]: ...
    async def create_ollama_model(self, modelfile: Path | str) -> dict[str, Any]: ...
    async def get_embeddings(self, input_text: str | list[str]) -> list[float] | list[list[float]]: ...
