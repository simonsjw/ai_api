from typing import TypedDict

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
