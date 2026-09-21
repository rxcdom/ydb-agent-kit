"""Test doubles for the tasks repositories: a session pool, a transaction, a migration loader."""
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
    """Records every statement and answers each one with the scripted rows."""

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
        """The last statement with its whitespace collapsed, so layout never matters."""
        return one_line(self.calls[-1][0])

    @property
    def last_parameters(self) -> Dict[str, Any]:
        return self.calls[-1][1]

    def last_values(self) -> Dict[str, Any]:
        """The plain values of the last statement's typed parameters."""
        return {name: typed.value for name, typed in self.last_parameters.items()}


class FakeTx:
    """A transaction context: records statements and streams no rows back."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any], bool]] = []
        self.rolled_back = False

    async def execute(self, query: str, parameters: Optional[dict] = None, commit_tx=False):
        self.calls.append((one_line(query), parameters or {}, commit_tx))

        async def stream():
            yield FakeResultSet([])

        return stream()

    async def rollback(self) -> None:
        self.rolled_back = True


class TransactionalFakePool(FakePool):
    """A pool whose transaction helper runs the callback once on one fake context."""

    def __init__(self) -> None:
        super().__init__()
        self.tx = FakeTx()

    async def retry_tx_async(self, callee):
        return await callee(self.tx)


def one_line(query: str) -> str:
    return " ".join(query.split())


def load_tasks_migration(version: str) -> Migration:
    """Load a migration of the tasks module the way the framework does."""
    files = {
        migration_file.version: migration_file
        for migration_file in discover_migration_files()
        if migration_file.module_name == "tasks"
    }
    return files[version].import_class()()


def declared_columns(migration: Migration, table: str) -> Dict[str, str]:
    """``{column: type}`` a migration declares for one table."""
    return {
        column: type_name
        for declared_table, column, type_name in migration.get_artifacts()["columns"]
        if declared_table == table
    }


def mapper_columns(mapper: Any) -> Dict[str, str]:
    """``{column: type}`` a mapper writes, in the spelling migrations use."""
    type_map = mapper.get_ydb_type_map()
    return {
        parameter.removeprefix("$"): format_ydb_type(type_map[parameter])
        for parameter in mapper.get_column_params()
    }


def assert_ddl_declares(statement: str, columns: Dict[str, str]) -> None:
    """Required columns are ``NOT NULL`` in the DDL; nullable ones are spelled with ``?``."""
    for column, type_name in columns.items():
        if type_name.endswith("?"):
            assert f" {column} {type_name}," in statement, f"{column} must be nullable"
        else:
            assert f" {column} {type_name} NOT NULL," in statement, f"{column} must be NOT NULL"
