import asyncio

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop to prevent asyncpg loop mismatch errors."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()
