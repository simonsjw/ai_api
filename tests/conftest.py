import asyncio

import pytest


@pytest.fixture(scope="session")
def event_loop_policy():
    """Ensure all tests share the same event loop (required for asyncpg pools)."""
    return asyncio.DefaultEventLoopPolicy()
