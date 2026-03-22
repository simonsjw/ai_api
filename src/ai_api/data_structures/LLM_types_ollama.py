"""
Ollama input types.

Simple immutable wrapper that matches the exact API of GrokInput
(from_list, frozen dataclass, etc.) so all tests work without changes.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OllamaInput:
    """Immutable input for Ollama requests (identical interface to GrokInput)."""

    messages: tuple[dict[str, Any], ...]

    @classmethod
    def from_list(cls, messages: list[dict[str, Any]]) -> "OllamaInput":
        """Create from a list of messages (exactly like GrokInput)."""
        return cls(messages=tuple(messages))

    def to_list(self) -> list[dict[str, Any]]:
        """Required by LLMRequest.to_dict() — matches GrokInput exactly."""
        return list(self.messages)

    def to_dict(self) -> dict[str, Any]:
        """For payload building and compatibility."""
        return {"messages": self.to_list()}
