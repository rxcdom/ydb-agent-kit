from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import patch

import pytest
import ydb

from src.shared.domain.migration import MigrationRecord
from src.shared.infrastructure.database.migration.base import Migration
from src.shared.infrastructure.database.migration.discovery import MigrationFile
from src.shared.infrastructure.database.migration.verification import (
    MigrationVerificationError,
    verify_all_migrations_completed,
    verify_migration_artifacts,
)

_FILE = MigrationFile(
    path=Path("/fake/notes/20260101000000_create_notes.py"),
    module_name="notes",
    version="20260101000000",
)


def _table(columns: Dict[str, object], indexes: List[str]) -> SimpleNamespace:
    return SimpleNamespace(
        columns=[ydb.Column(name, column_type) for name, column_type in columns.items()],
        indexes=[ydb.TableIndex(name) for name in indexes],
    )


class _FakeSession:
    def __init__(self, driver: "_FakeDriver") -> None:
        self._driver = driver

    async def create(self) -> "_FakeSession":
        self._driver.sessions_created += 1
        return self

    async def describe_table(self, path: str):
        self._driver.described_paths.append(path)
        if path in self._driver.broken_paths:
            raise ydb.issues.Unavailable("table service is unavailable")
        if path not in self._driver.tables:
            raise ydb.issues.SchemeError("")
        return self._driver.tables[path]

    async def delete(self) -> None:
        self._driver.sessions_deleted += 1


class _FakeDriver:
    """Stands in for the Table Service: absolute path -> described table."""

    def __init__(self, tables: Dict[str, SimpleNamespace], broken_paths=()) -> None:
        self.tables = tables
        self.broken_paths = set(broken_paths)
        self.described_paths: List[str] = []
        self.sessions_created = 0
        self.sessions_deleted = 0
        self.table_client = SimpleNamespace(session=lambda: _FakeSession(self))


class _DeclaringMigration(Migration):
    version = "20260101000000"
    description = "declares artifacts"

    def __init__(self, artifacts) -> None:
        self._artifacts = artifacts

    async def up(self, pool) -> None:
        return None

    def get_artifacts(self):
        return self._artifacts


def _notes_driver() -> _FakeDriver:
    return _FakeDriver(
        {
            "/local/notes": _table(
                {
                    "note_id": ydb.PrimitiveType.Utf8,
                    "body": ydb.OptionalType(ydb.PrimitiveType.Utf8),
                    "created_at": ydb.PrimitiveType.Timestamp,
                },
                indexes=["idx_notes_created"],
            )
        }
    )


async def test_matching_declaration_passes_and_describes_each_table_once() -> None:
    driver = _notes_driver()
    migration = _DeclaringMigration(
        {
            "tables": ["notes"],
            "indexes": [("notes", "idx_notes_created")],
            "columns": [
                ("notes", "note_id", "Utf8"),
                ("notes", "body", "Optional<Utf8>"),
                ("notes", "created_at"),
            ],
        }
    )

    await verify_migration_artifacts(driver, "/local", migration, _FILE)

    assert driver.described_paths == ["/local/notes"]
    assert driver.sessions_created == driver.sessions_deleted == 1


async def test_migration_without_artifacts_touches_nothing() -> None:
    driver = _notes_driver()

    await verify_migration_artifacts(driver, "/local", _DeclaringMigration({}), _FILE)

    assert driver.described_paths == []


async def test_every_problem_is_reported_in_one_error() -> None:
    driver = _notes_driver()
    migration = _DeclaringMigration(
        {
            "tables": ["notes", "tags"],
            "indexes": [("notes", "idx_notes_owner"), ("tags", "idx_tags_name")],
            "columns": [
                ("notes", "title"),
                ("notes", "created_at", "Datetime"),
                ("tags", "tag_id", "Utf8"),
            ],
        }
    )

    with pytest.raises(MigrationVerificationError) as raised:
        await verify_migration_artifacts(driver, "/local", migration, _FILE)

    message = str(raised.value)
    assert "Migration verification failed (6 errors)" in message
    assert "notes:20260101000000 declared table 'tags' but it does not exist" in message
    assert "declared index 'idx_notes_owner' on table 'notes' but it does not exist" in message
    assert "declared index 'idx_tags_name' on table 'tags' but the table does not exist" in message
    assert "declared column 'title' on table 'notes' but it does not exist" in message
    assert (
        "declared column 'created_at' of type 'Datetime' on table 'notes' "
        "but the actual type is 'Timestamp'" in message
    )
    assert "declared column 'tag_id' on table 'tags' but the table does not exist" in message


async def test_nullable_column_does_not_satisfy_a_required_type() -> None:
    driver = _notes_driver()
    migration = _DeclaringMigration({"columns": [("notes", "body", "Utf8")]})

    with pytest.raises(MigrationVerificationError) as raised:
        await verify_migration_artifacts(driver, "/local", migration, _FILE)

    message = str(raised.value)
    assert "(1 error)" in message
    assert "of type 'Utf8' on table 'notes' but the actual type is 'Utf8?'" in message
    assert "NOT NULL" in message


async def test_required_column_does_not_satisfy_a_nullable_type() -> None:
    driver = _notes_driver()
    migration = _DeclaringMigration({"columns": [("notes", "note_id", "Utf8?")]})

    with pytest.raises(MigrationVerificationError, match="the actual type is 'Utf8'"):
        await verify_migration_artifacts(driver, "/local", migration, _FILE)


async def test_malformed_declarations_are_reported_not_raised_as_crashes() -> None:
    driver = _notes_driver()
    migration = _DeclaringMigration(
        {
            "tables": "notes",
            "indexes": [("notes",), ("notes", 7)],
            "columns": [("notes", "note_id", "Utf8", "extra"), "notes.note_id"],
        }
    )

    with pytest.raises(MigrationVerificationError) as raised:
        await verify_migration_artifacts(driver, "/local", migration, _FILE)

    message = str(raised.value)
    assert "(5 errors)" in message
    assert "invalid 'tables' declaration: expected a list, got str" in message
    assert message.count("invalid index declaration") == 2
    assert message.count("invalid column declaration") == 2


async def test_artifacts_that_are_not_a_dictionary_are_rejected() -> None:
    migration = _DeclaringMigration(["notes"])

    with pytest.raises(MigrationVerificationError, match="must return a dictionary"):
        await verify_migration_artifacts(_notes_driver(), "/local", migration, _FILE)


async def test_schema_read_failure_is_reported_and_the_session_is_still_deleted() -> None:
    driver = _FakeDriver({}, broken_paths={"/local/notes"})
    migration = _DeclaringMigration({"tables": ["notes"]})

    with pytest.raises(MigrationVerificationError) as raised:
        await verify_migration_artifacts(driver, "/local", migration, _FILE)

    assert "cannot read the schema of table 'notes'" in str(raised.value)
    assert driver.sessions_created == driver.sessions_deleted == 1


async def test_unsafe_table_name_is_reported_without_reaching_the_database() -> None:
    driver = _notes_driver()
    migration = _DeclaringMigration({"tables": ["notes; DROP TABLE notes"]})

    with pytest.raises(MigrationVerificationError, match="Invalid table name"):
        await verify_migration_artifacts(driver, "/local", migration, _FILE)

    assert driver.described_paths == []


def _record(status: str, error_message=None) -> MigrationRecord:
    return MigrationRecord(
        version=_FILE.version,
        module_name=_FILE.module_name,
        description="",
        applied_at=dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc),
        checksum="abc",
        applied_by="local",
        status=status,
        error_message=error_message,
    )


_GET_APPLIED = (
    "src.shared.infrastructure.database.migration.verification.get_applied_migrations"
)


async def test_completed_and_unrecorded_migrations_pass_the_final_check() -> None:
    other = MigrationFile(path=Path("/fake/x.py"), module_name="notes", version="20260202000000")
    applied = {"notes:20260101000000": _record("completed")}

    with patch(_GET_APPLIED, return_value=applied):
        await verify_all_migrations_completed(object(), [_FILE, other])


@pytest.mark.parametrize(
    ("status", "error_message", "expected"),
    [
        ("failed", "boom", "has status 'failed'. Error: boom"),
        ("in_progress", None, "has status 'in_progress'"),
    ],
)
async def test_failed_or_unfinished_migration_blocks_the_final_check(
    status, error_message, expected
) -> None:
    applied = {"notes:20260101000000": _record(status, error_message)}

    with patch(_GET_APPLIED, return_value=applied):
        with pytest.raises(MigrationVerificationError, match=expected):
            await verify_all_migrations_completed(object(), [_FILE])
