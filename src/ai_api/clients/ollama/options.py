"""
Ollama hardware, runtime, and advanced options.

All fields are optional. Only non-None values are sent to the API.
Quantization hint belongs in LLMRequest.sys_spec (for future VRAM estimator).
"""

from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class OllamaOptions:
    """Typed wrapper for Ollama's native 'options' dict."""

    # ── Hardware / GPU allocation ─────────────────────────────────────
    num_gpu: int | None = None                                                            # -1 = all layers, 0 = CPU only, 999 = max possible
    num_thread: int | None = None                                                         # CPU threads for inference
    num_batch: int | None = None                                                          # batch size for processing
    main_gpu: int | None = None
    low_vram: bool | None = None
    flash_attn: bool | None = None                                                        # experimental faster attention

    # ── Context & memory ──────────────────────────────────────────────
    num_ctx: int | None = None                                                            # context window size
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    # ── Performance & cache ───────────────────────────────────────────
    use_mmap: bool | None = None
    use_mlock: bool | None = None
    kv_cache_type: str | None = None                                                      # "f16", "q8_0", "q4_0", etc.

    # ── Misc ──────────────────────────────────────────────────────────
    keep_alive: str | int | None = None                                                   # "5m", -1 = forever, 0 = unload immediately

    def to_dict(self) -> dict[str, Any]:
        """Return only non-None values (clean payload)."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
