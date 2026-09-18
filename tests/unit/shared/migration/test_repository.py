from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
import ydb

from src.shared.infrastructure.database.migration import repository


class _FakePool:
    def __init__(self, result=None, error=None) -> None:
        self._result = result if result is not None else []
        self._error = error
        self.calls = []

    async def execute_with_retries(self, query, parameters=None):
        self.calls.append((query, parameters))
        if self._error:
            raise self._error
        return self._result


def _row(**overrides) -> dict:
    row = {
        "module_name": "tasks",
        "version": "20260101000000",
        "description": "Create tasks",
        "applied_at": dt.datetime(2026, 9, 1, 12, 0),
        "checksum": "abc",
        "applied_by": "local",
        "status": "completed",
        "error_message": None,
    }
    row.update(overrides)
    return row


async def test_history_is_empty_while_the_table_does_not_exist() -> None:
    error = ydb.issues.SchemeError(
        "Cannot find table 'schema_migrations' because it does not exist"
    )

    assert await repository.get_applied_migrations(_FakePool(error=error)) == {}


async def test_other_errors_of_the_history_read_propagate() -> None:
    with pytest.raises(ydb.issues.SchemeError):
        await repository.get_applied_migrations(
            _FakePool(error=ydb.issues.SchemeError("Access denied"))
        )
    with pytest.raises(ydb.issues.Unavailable):
        await repository.get_applied_migrations(
            _FakePool(error=ydb.issues.Unavailable("does not exist right now"))
        )


async def test_rows_of_every_result_part_are_mapped_by_module_and_version() -> None:
    parts = [
        SimpleNamespace(index=0, rows=[_row()]),
        SimpleNamespace(
            index=0,
            rows=[_row(module_name="agent", version="20260201000000", status="failed",
                       error_message="index is missing")],
        ),
    ]

    applied = await repository.get_applied_migrations(_FakePool(result=parts))

    assert sorted(applied) == ["agent:20260201000000", "tasks:20260101000000"]
    completed = applied["tasks:20260101000000"]
    assert completed.applied_at == dt.datetime(2026, 9, 1, 12, 0, tzinfo=dt.timezone.utc)
    assert completed.error_message is None
    assert applied["agent:20260201000000"].error_message == "index is missing"


async def test_record_is_an_upsert_with_text_parameters_and_a_bounded_error() -> None:
    pool = _FakePool()

    await repository.record_migration(
        pool,
        version="20260101000000",
        module_name="tasks",
        description="Create tasks",
        checksum="abc",
        applied_by="local",
        status="failed",
        error_message="e" * 5000,
    )

    ((query, parameters),) = pool.calls
    assert "UPSERT INTO schema_migrations" in query
    assert "DECLARE $error_message AS Utf8?;" in query
    assert parameters["$module_name"].value == "tasks"
    assert parameters["$module_name"].value_type == ydb.PrimitiveType.Utf8
    assert parameters["$status"].value == "failed"
    assert parameters["$applied_at"].value.tzinfo is not None
    assert len(parameters["$error_message"].value) == repository.MAX_ERROR_MESSAGE_LENGTH


async def test_record_without_an_error_clears_the_previous_one() -> None:
    pool = _FakePool()

    await repository.record_migration(
        pool,
        version="20260101000000",
        module_name="tasks",
        description="Create tasks",
        checksum="abc",
        applied_by="local",
        status="completed",
    )

    assert pool.calls[0][1]["$error_message"].value is None


async def test_history_table_failure_is_logged_and_propagates(capsys) -> None:
    with pytest.raises(ydb.issues.Unavailable):
        await repository.ensure_migrations_table(
            _FakePool(error=ydb.issues.Unavailable("no storage pool"))
        )

    assert "Could not create table 'schema_migrations'" in capsys.readouterr().err


async def test_history_table_layout_uses_text_columns() -> None:
    pool = _FakePool()

    await repository.ensure_migrations_table(pool)

    query = pool.calls[0][0]
    assert "CREATE TABLE IF NOT EXISTS schema_migrations" in query
    assert "String" not in query
    assert "error_message Utf8?" in query
    assert "PRIMARY KEY (module_name, version)" in query
