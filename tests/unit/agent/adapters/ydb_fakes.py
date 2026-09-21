"""Test doubles shared by the agent repository tests: a pool, a transaction, a loader."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.shared.infrastructure.database.migration.base import Migration
from src.shared.infrastructure.database.migration.discovery import discover_migration_files
from src.shared.infrastructure.database.ydb.types import format_ydb_type


class FakeResultSet:
    def __init__(self, rows: List[Any]):
        self.rows = rows
        self.index = 0


class FakePool:
    """Records every statement and answers each one with the same rows."""

    def __init__(self, rows: Optional[List[Any]] = None, error: Optional[Exception] = None):
        self.rows = rows or []
        self.error = error
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    async def execute_with_retries(self, query: str, parameters: Optional[dict] = None):
        self.calls.append((query, parameters or {}))
        if self.error is not None:
            raise self.error
        return [FakeResultSet(self.rows)]

    @property
    def last_query(self) -> str:
        return self.calls[-1][0]

    @property
    def last_parameters(self) -> Dict[str, Any]:
        return self.calls[-1][1]


class FakeTx:
    """A transaction context: records statements and streams the scripted rows."""

    def __init__(self, rows: Optional[List[Any]] = None):
        self.rows = rows or []
        self.calls: List[Tuple[str, Dict[str, Any], bool]] = []

    async def execute(self, query: str, parameters: Optional[dict] = None, commit_tx=False):
        self.calls.append((query, parameters or {}, commit_tx))

        async def stream():
            yield FakeResultSet(self.rows)

        return stream()


def one_line(query: str) -> str:
    """Collapse whitespace so assertions do not depend on the query layout."""
    return " ".join(query.split())


def load_agent_migration(version: str) -> Migration:
    """Load a migration of the agent module the way the framework does."""
    files = {
        migration_file.version: migration_file
        for migration_file in discover_migration_files()
        if migration_file.module_name == "agent"
    }
    return files[version].import_class()()


def declared_columns(migration: Migration, table: str) -> Dict[str, str]:
    """Return ``{column: type}`` a migration declares for one table."""
    return {
        column: type_name
        for declared_table, column, type_name in migration.get_artifacts()["columns"]
        if declared_table == table
    }


def mapper_columns(mapper: Any) -> Dict[str, str]:
    """Return ``{column: type}`` a mapper writes, in the spelling migrations use."""
    type_map = mapper.get_ydb_type_map()
    return {
        parameter.removeprefix("$"): format_ydb_type(type_map[parameter])
        for parameter in mapper.get_column_params()
    }
