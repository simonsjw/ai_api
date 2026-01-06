import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

class OPEN_AI_MESSAGE(TypedDict):
    role: str
    content: str
    refusal: str | None

class OPEN_AI_CHOICE(TypedDict):
    index: int
    message: OPEN_AI_MESSAGE
    logprobs: int | None
    finish_reason: str

class OPEN_AI_USAGE(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: dict[str, int]

class OPEN_AI_BODY(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[OPEN_AI_CHOICE]
    usage: OPEN_AI_USAGE
    system_fingerprint: str

class OPEN_AI_RESPONSE(TypedDict):
    status_code: int
    request_id: str
    body: OPEN_AI_BODY

class OPEN_AI_PROMPT_OUTPUT(TypedDict):
    id: str
    custom_id: str
    response: OPEN_AI_RESPONSE
    error: str | None
type OPEN_AI_BATCH_OUTPUT = list[OPEN_AI_PROMPT_OUTPUT]
type SaveMode = Literal['none', 'json_files', 'postgres']
type Role = Literal['system', 'user', 'assistant']

class GrokMessage(TypedDict):
    role: Role
    content: str

@dataclass(frozen=True)
class GrokRequest:
    messages: list[GrokMessage]
    model: str = ...
    temperature: float = ...
    max_tokens: int | None = ...
    user_id: str | None = ...
    metadata: dict[str, Any] | None = ...
    request_id: uuid.UUID | None = field(default=None)
    def to_dict(self) -> dict[str, Any]: ...

@dataclass(frozen=True)
class GrokResponse:
    request: GrokRequest
    content: str
    response_id: str
    finish_reason: str
    usage: dict[str, Any]
    raw: dict[str, Any]
