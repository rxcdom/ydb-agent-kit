from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
import ydb

from src.shared.infrastructure.database.migration import operations
from src.shared.infrastructure.database.ydb.connection import YDBConnection


class _FakeSchema:
    """A mutable table description served through the Table Service double."""

    def __init__(self, tables: Dict[str, SimpleNamespace]) -> None:
        self.tables = tables
        self.table_client = SimpleNamespace(session=lambda: _FakeSession(self))


class _FakeSession:
    def __init__(self, schema: _FakeSchema) -> None:
        self._schema = schema

    async def create(self) -> "_FakeSession":
        return self

    async def describe_table(self, path: str):
        if path not in self._schema.tables:
            raise ydb.issues.SchemeError("")
        return self._schema.tables[path]

    async def delete(self) -> None:
        return None


class _FakePool:
    def __init__(self, on_execute=None, error: Optional[Exception] = None) -> None:
        self.queries: List[str] = []
        self._on_execute = on_execute
        self._error = error

    async def execute_with_retries(self, query: str, parameters=None):
        self.queries.append(query)
        if self._on_execute:
            self._on_execute()
        if self._error:
            raise self._error
        return []


def _table(columns: Dict[str, object], indexes=()) -> SimpleNamespace:
    return SimpleNamespace(
        columns=[ydb.Column(name, column_type) for name, column_type in columns.items()],
        indexes=[ydb.TableIndex(name) for name in indexes],
    )


def _connection(schema: _FakeSchema, pool: _FakePool) -> YDBConnection:
    return YDBConnection(driver=schema, pool=pool, database="/local")


_OPTIONAL_INT = ydb.OptionalType(ydb.PrimitiveType.Int32)


async def test_existing_index_is_not_created_again() -> None:
    schema = _FakeSchema({"/local/notes": _table({"id": ydb.PrimitiveType.Utf8}, ["idx_a"])})
    pool = _FakePool()

    await operations.add_index_if_not_exists(
        _connection(schema, pool), "notes", "idx_a", "ALTER TABLE notes ADD INDEX idx_a ..."
    )

    assert pool.queries == []


async def test_missing_index_is_created_with_the_given_statement() -> None:
    schema = _FakeSchema({"/local/notes": _table({"id": ydb.PrimitiveType.Utf8})})
    pool = _FakePool()

    await operations.add_index_if_not_exists(
        _connection(schema, pool), "notes", "idx_a", "ALTER TABLE notes ADD INDEX idx_a ..."
    )

    assert pool.queries == ["ALTER TABLE notes ADD INDEX idx_a ..."]


async def test_index_created_by_a_concurrent_process_is_accepted() -> None:
    schema = _FakeSchema({"/local/notes": _table({"id": ydb.PrimitiveType.Utf8})})
    pool = _FakePool(error=ydb.issues.BadRequest("error: path exist, request doesn't accept it"))

    await operations.add_index_if_not_exists(
        _connection(schema, pool), "notes", "idx_a", "ALTER TABLE ..."
    )


async def test_other_index_errors_propagate() -> None:
    schema = _FakeSchema({"/local/notes": _table({"id": ydb.PrimitiveType.Utf8})})
    pool = _FakePool(error=ydb.issues.Unavailable("try later"))

    with pytest.raises(ydb.issues.Unavailable):
        await operations.add_index_if_not_exists(
            _connection(schema, pool), "notes", "idx_a", "ALTER TABLE ..."
        )


async def test_missing_column_is_added_with_a_generated_statement() -> None:
    table = _table({"id": ydb.PrimitiveType.Utf8})
    schema = _FakeSchema({"/local/notes": table})
    pool = _FakePool(on_execute=lambda: table.columns.append(ydb.Column("score", _OPTIONAL_INT)))

    await operations.add_column_if_not_exists(
        _connection(schema, pool), "notes", "score", "Int32?"
    )

    assert pool.queries == ["ALTER TABLE `notes` ADD COLUMN `score` Int32?;"]


async def test_existing_column_of_the_same_type_is_left_alone() -> None:
    schema = _FakeSchema(
        {"/local/notes": _table({"id": ydb.PrimitiveType.Utf8, "score": _OPTIONAL_INT})}
    )
    pool = _FakePool()

    await operations.add_column_if_not_exists(
        _connection(schema, pool), "notes", "score", "Optional<Int32>"
    )

    assert pool.queries == []


async def test_existing_column_of_another_type_is_an_error() -> None:
    schema = _FakeSchema(
        {"/local/notes": _table({"id": ydb.PrimitiveType.Utf8, "score": _OPTIONAL_INT})}
    )
    pool = _FakePool()

    with pytest.raises(operations.SchemaMismatchError, match="has type 'Int32\\?'"):
        await operations.add_column_if_not_exists(
            _connection(schema, pool), "notes", "score", "Utf8?"
        )

    assert pool.queries == []


async def test_column_added_with_a_type_other_than_declared_is_an_error() -> None:
    table = _table({"id": ydb.PrimitiveType.Utf8})
    schema = _FakeSchema({"/local/notes": table})
    pool = _FakePool(on_execute=lambda: table.columns.append(ydb.Column("score", _OPTIONAL_INT)))

    with pytest.raises(operations.SchemaMismatchError, match="expected 'Int32'"):
        await operations.add_column_if_not_exists(
            _connection(schema, pool), "notes", "score", "Int32"
        )


async def test_existing_column_is_dropped_and_an_absent_one_is_skipped() -> None:
    schema = _FakeSchema(
        {"/local/notes": _table({"id": ydb.PrimitiveType.Utf8, "score": _OPTIONAL_INT})}
    )
    pool = _FakePool()
    connection = _connection(schema, pool)

    await operations.drop_column_if_exists(connection, "notes", "score")
    await operations.drop_column_if_exists(connection, "notes", "never_there")

    assert pool.queries == ["ALTER TABLE `notes` DROP COLUMN `score`;"]


async def test_dropping_a_column_of_a_missing_table_is_an_error() -> None:
    with pytest.raises(LookupError, match="table 'notes' does not exist"):
        await operations.drop_column_if_exists(
            _connection(_FakeSchema({}), _FakePool()), "notes", "score"
        )


@pytest.mark.parametrize("name", ["", "notes; DROP TABLE x", "1st", "na-me", "a.b"])
async def test_unsafe_identifiers_never_reach_the_database(name: str) -> None:
    pool = _FakePool()
    connection = _connection(_FakeSchema({}), pool)

    with pytest.raises(ValueError, match="Invalid"):
        await operations.drop_column_if_exists(connection, "notes", name)
    with pytest.raises(ValueError, match="Invalid"):
        await operations.add_column_if_not_exists(connection, name, "score", "Int32?")

    assert pool.queries == []
