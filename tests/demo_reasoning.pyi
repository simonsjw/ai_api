from typing import Any

env_var: str | None
db_settings: dict[str, str | list[str]]
XAI_API_KEY: str

async def run_demo(demo_logger: Logger, provider: str, model: str, question: str, include_reasoning: bool, reasoning_effort: str | None = None, api_key: str | None = None, settings: dict[str, Any] | None = None) -> None: ...
async def main() -> None: ...
