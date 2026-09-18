"""Shared fixtures of the integration suite.

Integration tests talk to a real YDB. They run only when ``YDB_ENDPOINT`` is
set in the environment, for example::

    YDB_ENDPOINT=grpc://localhost:2136 YDB_DISABLE_DISCOVERY=true \
        python -m pytest tests/integration -m integration

The database may be shared, so a test creates only uniquely named objects and
removes them when it finishes.
"""
from __future__ import annotations

import os
from typing import AsyncIterator

import pytest

from src.shared.infrastructure.database.ydb.connection import (
    YDBConnection,
    open_ydb_connection,
)
from src.shared.infrastructure.database.ydb.settings import YDBSettings

ENDPOINT_VARIABLE = "YDB_ENDPOINT"
SKIP_REASON = f"integration tests need a running YDB: set {ENDPOINT_VARIABLE}"


def ydb_is_configured() -> bool:
    return bool(os.environ.get(ENDPOINT_VARIABLE, "").strip())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if ydb_is_configured():
        return
    skip = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        if item.get_closest_marker("integration") is not None:
            item.add_marker(skip)


@pytest.fixture
async def ydb_connection() -> AsyncIterator[YDBConnection]:
    """An open connection built from the environment, closed after the test."""
    if not ydb_is_configured():
        pytest.skip(SKIP_REASON)
    connection = await open_ydb_connection(YDBSettings.from_env())
    try:
        yield connection
    finally:
        await connection.close()
