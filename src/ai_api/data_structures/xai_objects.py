"""
xAI API data models for the ai_api package.

This module defines the complete, type-safe set of data structures required to
interact with the xAI API. It provides a clean abstraction layer for both
single-request and batch operations, streaming responses, persistence
configuration, and message construction.

All public classes, protocols, and enumerations are exported via ``__all__``
and are designed to work seamlessly with ``xAIClient`` (the primary entry point
in ``core/grok_client.py``).

Public exports (defined in ``__all__``):

1. **Request Hierarchy** (xAIMessage → xAIInput → xAIRequest)
   - ``xAIMessage``: The fundamental building block of a conversation. It
     represents a single message with a ``role`` (from the ``Role`` enum) and
     ``content`` (text or multimodal parts).
   - ``xAIInput``: A flexible container that accepts either a simple ``str``
     (for quick text-only prompts) or a list of ``xAIMessage`` objects (for
     system prompts, multimodal content, tools, etc.).
   - ``xAIRequest``: The top-level request model. It composes a ``xAIInput``
     (or simple string) with API parameters (``model``, ``temperature``,
     ``max_tokens``, ``include_reasoning``, etc.) and configuration options
     such as ``save_mode`` and ``prompt_cache_key``. All calls to
     ``xAIClient.generate()`` begin with a ``xAIRequest``.

2. **Batch Processing**
   - ``xAIBatchRequest``: Groups multiple ``xAIRequest`` objects into a single
     batch submission. It reduces API overhead and cost when processing many
     prompts together and provides batch-level metadata (``batch_id``,
     ``batch_index``, etc.).
   - ``xAIBatchResponse``: The corresponding response object that aggregates
     the individual results of each request in a completed batch.

3. **Response and Streaming Models**
   - ``xAIResponse``: Represents a single successful response from the xAI
     API. It is instantiated exclusively via the factory method
     ``xAIResponse.from_dict()`` and provides symmetric helper methods to
     ``xAIRequest`` (``get_messages()``, ``extract_response_snippet()``,
     ``get_reasoning_content()``, ``has_media()``).
   - ``xAIStreamingChunk``: Concrete implementation of a single chunk of
     streamed output received during a streaming generation.
   - ``LLMStreamingChunkProtocol``: A structural Protocol (using ``typing.Protocol``)
     that ``xAIStreamingChunk`` implements. It defines the common interface
     expected by streaming consumers across different LLM providers in the
     ``ai_api`` package, enabling polymorphic handling of streaming results.

4. **Supporting Types**
   - ``SaveMode``: Enumeration that controls persistence behaviour for a
     request/response pair. Valid values are typically ``"none"`` (no
     persistence) and ``"postgres"`` (full request and response logging to the
     PostgreSQL database via ``_persist_request()`` and ``_persist_response()``
     in ``xAIClient``).
   - ``Role``: Enumeration of standard message roles (``system``, ``user``,
     ``assistant``, ``tool``, etc.) used consistently by ``xAIMessage`` and
     when constructing SDK messages.

5. **Design principles**:
   - The request hierarchy ensures flexible input handling while preventing brittle
     direct attribute access (all downstream code must use the public helper
     methods on ``xAIRequest`` and ``xAIResponse``).
   - Input-side media saving now occurs at request time; response-side helpers
     remain focused on output processing.
   - Streaming and batch support maintain full compatibility with the core
     ``xAIClient.generate()`` and ``xAIClient`` batch methods.

These models are the foundation for all xAI-specific functionality in the
package. Application code should interact primarily with ``xAIRequest`` and
``xAIResponse``; the lower-level models provide internal structure and
extensibility.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, Sequence, Type, cast, runtime_checkable

from pydantic import BaseModel

__all__: list[str] = [
    "xAIMessage",
    "xAIInput",
    "xAIRequest",
    "xAIBatchRequest",
    "xAIResponse",
    "xAIBatchResponse",
    "xAIStreamingChunk",
    "LLMStreamingChunkProtocol",
    "SaveMode",
    "Role",
    "xAIJSONResponseSpec",
    "JSON_INSTRUCTION",
]

type SaveMode = Literal["none", "json_files", "postgres"]
type Role = Literal["system", "user", "assistant", "developer"]

# Required system-prompt substring that xAI expects when JSON mode is active.
# The check is performed on the full content string (case-sensitive).
JSON_INSTRUCTION: str = "Extract the requested information as structured JSON."


@runtime_checkable
class LLMStreamingChunkProtocol(Protocol):
    """Common contract for streaming chunks across providers."""

    @property
    def text(self) -> str: ...
    @property
    def finish_reason(self) -> str | None: ...
    @property
    def tool_calls_delta(self) -> list[dict[str, Any]] | None: ...
    @property
    def is_final(self) -> bool: ...
    @property
    def raw(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class xAIJSONResponseSpec:
    """Specification for enforcing JSON-structured responses.

    Accepts either a Pydantic ``BaseModel`` subclass (recommended) or a raw
    JSON-schema dictionary. When attached to an ``xAIRequest``, the underlying
    xAI SDK automatically sets the appropriate ``response_format`` and the
    request validator ensures the system prompt contains the required
    instruction.

    Parameters
    ----------
    model : type[BaseModel] | dict[str, Any] | None
        Pydantic model class or raw JSON schema. ``None`` disables structured
        output.
    """

    model: type[BaseModel] | dict[str, Any] | None = None

    def to_sdk_response_format(self) -> dict[str, Any] | None:
        """Convert the spec to the exact format expected by xAI SDK's
        ``chat.create()`` (or ``responses`` endpoint).

        Returns
        -------
        dict[str, Any] | None
            SDK-ready ``response_format`` dict or ``None`` if no spec.
        """
        if self.model is None:
            return None
        if isinstance(self.model, dict):
            # Raw schema supplied by caller
            return {"type": "json_schema", "json_schema": self.model}
        # Pydantic BaseModel -> SDK auto-converts (native support)
        return self.model                                                                 # type: ignore[return-value]

    # The xAI SDK handles BaseModel passthrough directly.


@dataclass(frozen=True)
class xAIMessage:
    """Immutable message supporting text, images, and file attachments.

    Parameters
    ----------
    role : Role
        Message role.
    content : str | list[dict[str, Any]]
        Text string or multimodal content list.
    """

    role: Role
    content: str | list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert the message to a dictionary for JSON serialisation.

        Returns
        -------
        dict[str, Any]
            Serialisable representation.
        """
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, msg_dict: dict[str, Any]) -> "xAIMessage":
        """Create xAIMessage from dict with full validation.

        Parameters
        ----------
        msg_dict : dict[str, Any]
            Dictionary with 'role' and 'content'.

        Returns
        -------
        xAIMessage
            Validated instance.

        Raises
        ------
        ValueError
            If keys missing, role invalid or content malformed.
        """
        cls._validate_keys(msg_dict)                                                      # early key check
        role = cls._validate_role(msg_dict["role"])
        content = msg_dict["content"]
        cls._validate_content(content)
        return cls(role=role, content=content)

    @staticmethod
    def _validate_keys(msg_dict: dict[str, Any]) -> None:
        """Validate required keys exist."""
        required_keys = {"role", "content"}
        if not required_keys.issubset(msg_dict.keys()):
            missing = required_keys - set(msg_dict.keys())
            raise ValueError(f"Missing keys: {missing}")

    @staticmethod
    def _validate_role(role_str: str) -> Role:
        """Validate and cast role string."""
        allowed_roles = ("system", "user", "assistant", "developer")
        if role_str not in allowed_roles:
            raise ValueError(
                f"Invalid role '{role_str}'. Must be one of: {allowed_roles}"
            )
        return cast(Role, role_str)

    @staticmethod
    def _validate_content(content: Any) -> None:
        """Validate content type and structure."""
        if isinstance(content, str):
            return
        if not isinstance(content, list):
            raise ValueError("Content must be str or list[dict].")
        for item in content:
            xAIMessage._validate_content_item(item)

    @staticmethod
    def _validate_content_item(item: dict[str, Any]) -> None:
        """Validate a single multimodal item."""
        if not isinstance(item, dict):
            raise ValueError("List items in content must be dicts.")
        typ = item.get("type")
        if typ == "input_text":
            if "text" not in item or not isinstance(item["text"], str):
                raise ValueError("input_text requires 'text': str.")
        elif typ == "input_image":
            if "image_url" not in item or not isinstance(item["image_url"], str):
                raise ValueError("input_image requires 'image_url': str.")
            detail = item.get("detail", "auto")
            if detail not in ("auto", "low", "high"):
                raise ValueError("detail must be 'auto', 'low', or 'high'.")
        elif typ is None:
            raise ValueError("Each item must have 'type' key.")
        else:
            raise ValueError(f"Invalid type '{typ}' in content item.")


@dataclass(frozen=True)
class xAIInput:
    """Wrapper for a sequence of xAIMessage instances (native 'input')."""

    messages: tuple[xAIMessage, ...] = field(default_factory=tuple)

    def to_list(self) -> list[dict[str, Any]]:
        """Native list-of-dict representation for xAI SDK.

        Returns
        -------
        list[dict[str, Any]]
            List of message dictionaries.
        """
        return [msg.to_dict() for msg in self.messages]

    @classmethod
    def from_str(cls, text: str) -> "xAIInput":
        """Convenience factory for single-user-message string input.

        Parameters
        ----------
        text : str
            User message text.

        Returns
        -------
        xAIInput
            Instance containing one user message.
        """
        return cls(messages=(xAIMessage(role="user", content=text),))

    @classmethod
    def from_list(
        cls, messages: Sequence[dict[str, Any] | xAIMessage] | None = None
    ) -> "xAIInput":
        """Create xAIInput from sequence of dicts or xAIMessage objects.

        Parameters
        ----------
        messages : Sequence[dict[str, Any] | xAIMessage] | None
            Input messages.

        Returns
        -------
        xAIInput
            Validated input wrapper.
        """
        if not messages:
            return cls(messages=tuple())
        processed: list[xAIMessage] = []
        for msg in messages:
            if isinstance(msg, xAIMessage):
                processed.append(msg)
            elif isinstance(msg, dict):
                processed.append(xAIMessage.from_dict(msg))
            else:
                raise TypeError(
                    f"Expected dict or xAIMessage, got {type(msg).__name__}"
                )
        return cls(messages=tuple(processed))


@dataclass(frozen=True)
class xAIRequest:
    """Represents a request to the xAI API.

    The model supports two input styles for maximum flexibility:
    - Simple text: ``input="Explain prompt caching..."`` (most common)
    - Structured messages: ``input=List[dict]`` or a messages container
      (required for system prompts, multimodal content, etc.)

    All downstream code (including persistence, media handling, and the XAI SDK
    integration in xAIClient) MUST use the public helper methods rather than
    accessing ``.input`` directly. This eliminates the previous design
    inconsistency that caused AttributeError on string inputs.

    When ``response_spec`` is supplied the request automatically:
      1. Configures the SDK ``response_format``.
      2. Validates that at least one system message contains the required
         JSON-instruction string.

    **Public instance methods** (all pure and deterministic):
        - ``to_sdk_messages()``: Returns normalised list of message dicts for
          the XAI SDK (existing method – unchanged).
        - ``get_messages() -> list[dict[str, Any]]``: Returns normalised
          messages in the same format as ``to_sdk_messages()``. Always safe
          for both simple ``str`` and complex input.
        - ``extract_prompt_snippet(max_chars: int = 100) -> str``: Returns the
          first 100 characters of the first user message (or joined text parts
          for multimodal). Used for prompt_snippet in requests.meta.
        - ``has_media() -> bool``: Returns True if any user message contains
          images or files. Used to decide whether media-saving logic runs.

    **Intended use**:
        - Instantiate once per API call.
        - Pass to ``xAIClient.generate(request, ...)``.
        - Media files (input-side only) are now saved during ``_persist_request``
          (not response time) to fail-fast and keep persistence concerns
          logically separated.
        - The ``save_mode="postgres"`` path in xAIClient relies on these
          helpers for clean meta storage.

    Parameters
    ----------
    input : xAIInput | str
        Prompt content (string or full message list).
    model : str
        xAI model identifier.
    temperature : float
        Sampling temperature (0.0–2.0).
    top_p: float | None
        The percentile to which to limit selection.
    max_tokens : int | None
        Maximum output tokens.
    response_spec : xAIJSONResponseSpec | None
        Structured-output specification (replaces old response_format).
    include_reasoning : bool
        Whether to return Grok reasoning trace.
    reasoning_effort : Literal[str]
        The reasoning effort to expend on a prompt.
        This parameter is not supported by grok-4.20
        In multi agent models, it selects the number agents to be used.
    tools: list[dict]
        Tools to be used by the model.
    save_mode : SaveMode
        Persistence behaviour.
    prompt_cache_key : str | None
        Optional key for prompt-caching server affinity.
    batch_request_id: str | None

    """

    input: "xAIInput" | str                                                               # forward ref
    model: str
    temperature: float | None = None
    top_p: float | None = None
    logprobs: bool | None = None                                                          # not implemented
    top_logprobs: int | None = None                                                       # not implemented
    max_tokens: int | None = None
    response_spec: xAIJSONResponseSpec | None = None
    include_reasoning: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    tools: list[dict[str, Any]] | None = None
    save_mode: SaveMode = "none"
    prompt_cache_key: str | None = None
    batch_request_id: str | None = None

    def __post_init__(self) -> None:
        """Perform post-initialisation validation and automatic configuration.

        If a response_spec is present we:
          - set the internal SDK response_format (via helper)
          - enforce the JSON-instruction in the system prompt.
        """
        # Inline comment: automatic JSON enforcement when spec supplied
        if self.response_spec is not None:
            self._validate_json_instruction_present()
            # The helper below is called by to_chat_create_kwargs()

    def _validate_json_instruction_present(self) -> None:
        """Raise if no system message contains the required JSON instruction.

        The check scans the rendered message list (case-sensitive substring).
        """
        messages: list[dict[str, Any]] = self.get_messages()                              # re-uses existing helper
        instruction_found = any(
            JSON_INSTRUCTION in msg["content"]
            for msg in messages
            if msg["role"] == "system" and isinstance(msg["content"], str)
        )
        if not instruction_found:
            raise ValueError(
                f"JSON response_spec supplied but no system message contains "
                f"the required instruction: '{JSON_INSTRUCTION}'"
            )

    def to_sdk_messages(self) -> list[dict[str, Any]]:
        """Native representation for xAI SDK chat.create / append.

        Normalises legacy str input to xAIInput (single user message)
        for backward compatibility with tests and direct construction.
        """
        if isinstance(self.input, str):                                                   # legacy convenience path
            return xAIInput.from_str(self.input).to_list()
        if isinstance(self.input, xAIInput):
            return self.input.to_list()
        # fallback for list-of-dict (rare)
        return self.input if isinstance(self.input, list) else []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "xAIRequest":
        """Convert generic input dict (may contain 'messages' or 'input') into typed native xAIRequest."""
        data = dict(data)
        if "messages" in data and "input" not in data:
            data["input"] = data.pop("messages")
        if isinstance(data.get("input"), (list, tuple)):
            data["input"] = xAIInput.from_list(data["input"])
        elif isinstance(data.get("input"), dict):
            data["input"] = xAIInput.from_list(data["input"].get("messages", []))
        return cls(**data)

    def get_messages(self) -> list[dict[str, Any]]:
        """Return normalised list of message dicts (role/content) for any input type.

        Reuses the existing to_sdk_messages() normalisation to guarantee consistency.
        """
        if isinstance(self.input, str):
            return [{"role": "user", "content": self.input}]
        # Complex input (object with .messages, list of dicts, etc.)
        messages = list(self.to_sdk_messages())
        return messages

    def to_chat_create_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for xAI SDK `chat.create()`.

        Centralises forwarding of all request parameters (tools, sampling
        controls, etc.) for efficiency and single source of truth.

        Returns
        -------
        dict[str, Any]
            Kwargs compatible with `AsyncClient.chat.create`.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "store_messages": True,                                                       # Enables Responses API + prompt caching
            "tools": self.tools,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "include_reasoning": self.include_reasoning,
            "reasoning_effort": self.reasoning_effort,
        }

        # Automatic response_format when spec is present
        if self.response_spec is not None:
            fmt = self.response_spec.to_sdk_response_format()
            if fmt is not None:
                kwargs["response_format"] = fmt

        return kwargs

    def extract_prompt_snippet(self, max_chars: int = 100) -> str:
        """Extract a truncated prompt snippet from the first user message.

        Handles both simple str input and complex message structures.
        """
        messages = self.get_messages()
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:max_chars]
                elif isinstance(content, list):
                    # For multimodal, join text parts only
                    text_parts: list[str] = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif (
                            isinstance(part, dict) and part.get("type") == "input_text"
                        ):
                            text_parts.append(part.get("text", ""))
                    snippet = " ".join(text_parts)
                    return snippet[:max_chars]
        return ""                                                                         # fallback for empty / system-only prompts

    def has_media(self) -> bool:
        """Return True if any user message contains multimodal content (images/files).

        Detects list-style content or dicts with image/file types.
        """
        messages = self.get_messages()
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype in ("input_image", "input_file"):
                            return True
                        # Future: extend for other media indicators if SDK evolves
        return False

    def with_updates(self, **updates: Any) -> "xAIRequest":
        """Type-safe replacement for object.__setattr__ (used by tests)."""
        return replace(self, **updates)


@dataclass(frozen=True)
class xAIBatchRequest:
    """Container for a batch of xAIRequests (used by client.batch.add)."""

    batch_name: str
    requests: list[xAIRequest] = field(default_factory=list)

    def to_sdk_batch_requests(self) -> list[Any]:
        """Will be converted by client to xAI SDK prepared objects."""
        return self.requests


@dataclass(frozen=True)
class xAIResponse:
    """Represents the response received from the xAI API.

    Created exclusively via the factory method ``from_dict()`` (which handles
    raw SDK output, extracts reasoning traces, and validates required fields).

    Provides symmetric helper methods to xAIRequest so that persistence,
    logging, and downstream consumers never perform raw dict traversal.

    **Public instance methods**:
        - ``get_messages() -> list[dict[str, Any]]``: Returns assistant message(s)
          (and optional reasoning messages) in normalised format.
        - ``extract_response_snippet(max_chars: int = 200) -> str``: Returns a
          truncated snippet of the main assistant content.
        - ``get_reasoning_content(max_chars: int = 500) -> str | None``: Returns
          the reasoning/thinking trace (when ``include_reasoning=True`` and
          supported by the model).
        - ``has_media() -> bool``: Always False for current text-only xAI
          responses; implemented for future image-generation or file-output
          capabilities.

    **Factory method**:
        - ``from_dict(cls, data: dict[str, Any]) -> xAIResponse``: Alternative
          constructor that parses the raw API dictionary, extracts ``id``,
          ``created_at``, ``model``, ``output``, ``usage``, ``status``, and
          any reasoning trace. Raises ValueError on missing required fields.

    **Intended use**:
        - Inside ``xAIClient._persist_response(...)``: instantiate with
          ``xAIResponse.from_dict(result)`` then call the helpers to populate
          ``responses.meta``.
        - Logging, auditing, or UI display layers should call the helpers
          rather than accessing ``.raw``, ``.output``, or ``.reasoning_text``
          directly.
        - Keeps the persistence layer clean and makes future extensions
          (e.g., when xAI returns generated media) trivial.
    """

    id: str
    created_at: int
    model: str
    output: Sequence[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    status: Literal["completed", "in_progress", "incomplete"] | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    reasoning_text: str | None = None

    @property
    def text(self) -> str:
        """Extract the main text content from the response."""
        parts: list[str] = []
        for item in self.output:
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    parts.append(part.get("text", ""))
        return "".join(parts).strip()

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Extract any tool calls from the response."""
        calls: list[dict[str, Any]] = []
        for item in self.output:
            t = item.get("type")
            if t in (
                "function_call",
                "web_search_call",
                "x_search_call",
                "code_interpreter_call",
                "file_search_call",
                "mcp_call",
            ):
                calls.append(item)
        return calls

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "xAIResponse":
        """Create a xAIResponse instance from a raw API response dictionary.

        Also extracts the native reasoning trace.
        """
        raw = data.copy()
        try:
            response_id = str(data["id"])
            created_at = int(data["created_at"])
            model = str(data["model"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Missing or invalid required field(s) in API response: "
                "'id', 'created_at', 'model'"
            ) from exc

        output = data.get("output", [])
        if not isinstance(output, list):
            output = []

        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            usage = None

        status = data.get("status")
        if status not in ("completed", "in_progress", "incomplete", None):
            status = None

            # Extract reasoning trace (model-dependent)
        reasoning_text: str | None = None
        for item in data.get("output", []):
            if item.get("type") == "message":
                reasoning_text = item.get("reasoning_content")
                break
            if isinstance(item.get("reasoning"), dict):
                reasoning_text = item.get("reasoning", {}).get("encrypted_content")
                break

        return cls(
            id=response_id,
            created_at=created_at,
            model=model,
            output=output,
            usage=usage,
            status=status,
            raw=raw,
            reasoning_text=reasoning_text,
        )

    def __str__(self) -> str:
        txt = self.text[:80] + "…" if len(self.text) > 80 else self.text
        tc = len(self.tool_calls)
        return f"xAIResponse(id={self.id}, model={self.model}, text={txt!r}, tool_calls={tc})"

    def get_messages(self) -> list[dict[str, Any]]:
        """Return the assistant message(s) from the xAI response in normalised format.

        Mirrors xAIRequest.get_messages() for consistent API usage across request/response objects.
        """
        messages: list[dict[str, Any]] = []
        try:
            # xAI SDK response follows OpenAI-compatible structure
            raw = getattr(
                self, "raw", getattr(self, "response", getattr(self, "data", {}))
            )
            choices = raw.get("choices", []) or raw.get("output", {}).get("choices", [])
            for choice in choices:
                message = choice.get("message") or choice.get("delta", {})
                if message:
                    role = message.get("role", "assistant")
                    content = message.get("content")
                    messages.append({"role": role, "content": content})

                    # Include reasoning/thinking content when present (for supported models)
                    reasoning = message.get("reasoning_content") or message.get(
                        "reasoning"
                    )
                    if reasoning:
                        messages.append({"role": "reasoning", "content": reasoning})
        except (AttributeError, TypeError, KeyError):
            # Graceful fallback for unexpected or minimal response structures
            pass
        return messages

    def extract_response_snippet(self, max_chars: int = 200) -> str:
        """Extract a truncated snippet of the main assistant response text.

        Mirrors xAIRequest.extract_prompt_snippet() for consistent logging/persistence.
        """
        messages = self.get_messages()
        for msg in messages:
            if msg.get("role") in ("assistant", "reasoning"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:max_chars]
                # Final fallback: direct raw lookup (common in xAI responses)
        try:
            raw = getattr(
                self, "raw", getattr(self, "response", getattr(self, "data", {}))
            )
            content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(content, str):
                return content[:max_chars]
        except (IndexError, AttributeError, TypeError):
            pass
        return ""

    def get_reasoning_content(self, max_chars: int = 500) -> str | None:
        """Return reasoning/thinking content if generated (for models that support include_reasoning)."""
        messages = self.get_messages()
        for msg in messages:
            if msg.get("role") == "reasoning":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:max_chars]
        return None

    def has_media(self) -> bool:
        """Return True if the response contains generated media (images, files, etc.).

        Currently always False for xAI text completions; implemented for future-proofing
        when image-generation or file-output capabilities are added to the xAI API.
        """
        # xAI-4 currently returns text-only output. Extend this method when needed.
        return False


@dataclass(frozen=True)
class xAIBatchResponse:
    """Immutable representation of a completed xAI batch result."""

    batch_id: str
    results: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "xAIBatchResponse":
        """Create xAIBatchResponse from the raw /v1/batches/{id} response."""
        return cls(
            batch_id=str(data["id"]),
            results=tuple(data.get("results", [])),
        )


@dataclass(frozen=True)
class xAIStreamingChunk:
    """Native streaming chunk (implements LLMStreamingChunkProtocol)."""

    text: str = ""
    finish_reason: str | None = None
    tool_calls_delta: list[dict[str, Any]] | None = None
    is_final: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
