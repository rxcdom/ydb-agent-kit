from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, List, Optional

import pytest
import ydb

from src.shared.infrastructure.database.migration import executor
from src.shared.infrastructure.database.migration.discovery import MigrationFile
from src.shared.infrastructure.database.migration.verification import (
    MigrationVerificationError,
)
from src.shared.infrastructure.database.ydb.connection import YDBConnection

_VERSION = "20260101000000"

_MIGRATION_SOURCE = '''
from src.shared.infrastructure.database.migration.base import Migration


class ScratchMigration(Migration):
    version = "20260101000000"
    description = "Create the scratch table"

    def get_artifacts(self):
        return {artifacts}

    async def up(self, pool):
        {body}
'''


class _FakePool:
    """Records history upserts as ``status:<value>`` and other queries verbatim."""

    def __init__(self, failing_status: Optional[str] = None) -> None:
        self.events: List[str] = []
        self.history_rows: List[dict] = []
        self._failing_status = failing_status

    async def execute_with_retries(self, query: str, parameters: Any = None) -> list:
        if parameters and "$status" in parameters:
            row = {name: typed.value for name, typed in parameters.items()}
            if row["$status"] == self._failing_status:
                raise ydb.issues.Unavailable("history table is unavailable")
            self.history_rows.append(row)
            self.events.append(f"status:{row['$status']}")
        else:
            self.events.append(query.strip())
        return []


def _connection(pool: _FakePool) -> YDBConnection:
    return YDBConnection(driver=object(), pool=pool, database="/local")


def _migration_file(tmp_path: Path, *, artifacts: str, body: str) -> MigrationFile:
    path = tmp_path / f"{_VERSION}_create_scratch.py"
    path.write_text(_MIGRATION_SOURCE.format(artifacts=artifacts, body=body), encoding="utf-8")
    return MigrationFile(path=path, module_name="scratch", version=_VERSION)


@pytest.fixture
def verification_calls(monkeypatch) -> List[tuple]:
    """Replace artifact verification with a recorder that passes."""
    calls: List[tuple] = []

    async def passing(driver, database, migration_instance, migration_file):
        calls.append((driver, database, migration_file.version))

    monkeypatch.setattr(executor, "verify_migration_artifacts", passing)
    return calls


async def test_successful_migration_goes_from_in_progress_to_completed(
    tmp_path, verification_calls
) -> None:
    pool = _FakePool()
    connection = _connection(pool)
    migration_file = _migration_file(
        tmp_path,
        artifacts="{'tables': ['scratch']}",
        body="await pool.execute_with_retries('CREATE TABLE scratch')",
    )

    await executor.run_migration(connection, migration_file, applied_by="tester")

    assert pool.events == ["status:in_progress", "CREATE TABLE scratch", "status:completed"]
    assert verification_calls == [(connection.driver, "/local", _VERSION)]

    completed = pool.history_rows[-1]
    assert completed["$module_name"] == "scratch"
    assert completed["$version"] == _VERSION
    assert completed["$description"] == "Create the scratch table"
    assert completed["$applied_by"] == "tester"
    assert completed["$checksum"] == hashlib.sha256(migration_file.path.read_bytes()).hexdigest()
    assert completed["$error_message"] is None


async def test_verification_is_skipped_when_nothing_is_declared(
    tmp_path, verification_calls
) -> None:
    pool = _FakePool()
    migration_file = _migration_file(tmp_path, artifacts="{}", body="return None")

    await executor.run_migration(_connection(pool), migration_file, applied_by="tester")

    assert verification_calls == []
    assert pool.events == ["status:in_progress", "status:completed"]


async def test_verification_failure_records_failed_and_raises(tmp_path, monkeypatch) -> None:
    async def failing(driver, database, migration_instance, migration_file):
        raise MigrationVerificationError("declared table 'scratch' but it does not exist")

    monkeypatch.setattr(executor, "verify_migration_artifacts", failing)
    pool = _FakePool()
    migration_file = _migration_file(
        tmp_path, artifacts="{'tables': ['scratch']}", body="return None"
    )

    with pytest.raises(MigrationVerificationError, match="does not exist"):
        await executor.run_migration(_connection(pool), migration_file, applied_by="tester")

    assert pool.events == ["status:in_progress", "status:failed"]
    error_message = pool.history_rows[-1]["$error_message"]
    assert error_message.startswith("MigrationVerificationError:")
    assert "declared table 'scratch'" in error_message


async def test_error_inside_up_records_failed_with_a_bounded_message(
    tmp_path, verification_calls
) -> None:
    pool = _FakePool()
    migration_file = _migration_file(
        tmp_path,
        artifacts="{'tables': ['scratch']}",
        body="raise RuntimeError('x' * 5000)",
    )

    with pytest.raises(RuntimeError):
        await executor.run_migration(_connection(pool), migration_file, applied_by="tester")

    assert pool.events == ["status:in_progress", "status:failed"]
    assert verification_calls == []
    assert len(pool.history_rows[-1]["$error_message"]) == 1000


async def test_original_error_survives_a_failure_to_record_the_failed_status(
    tmp_path, verification_calls, capsys
) -> None:
    pool = _FakePool(failing_status="failed")
    migration_file = _migration_file(
        tmp_path, artifacts="{}", body="raise RuntimeError('up() exploded')"
    )

    with pytest.raises(RuntimeError, match="exploded"):
        await executor.run_migration(_connection(pool), migration_file, applied_by="tester")

    assert pool.events == ["status:in_progress"]
    assert "Could not record the failed status" in capsys.readouterr().err
