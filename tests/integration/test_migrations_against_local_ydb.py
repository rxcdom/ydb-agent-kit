"""The migration framework against a real YDB.

Covers what fakes cannot: the Table Service session lifecycle, the way the
server reports column types and nullability, the history table, and the
idempotent ``ALTER TABLE`` helpers. Every object created here carries a unique
suffix and is removed afterwards, including the history rows.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, List

import pytest
import ydb

from src.shared.infrastructure.database.migration import operations, schema
from src.shared.infrastructure.database.migration.discovery import discover_migration_files
from src.shared.infrastructure.database.migration.manager import apply_pending_migrations
from src.shared.infrastructure.database.migration.planning import plan_migrations_to_run
from src.shared.infrastructure.database.migration.repository import (
    MIGRATIONS_TABLE,
    get_applied_migrations,
)
from src.shared.infrastructure.database.migration.verification import (
    MigrationVerificationError,
    verify_migration_artifacts,
)
from src.shared.infrastructure.database.ydb.connection import YDBConnection

pytestmark = pytest.mark.integration

APPLIED_BY = "integration-test"

_HONEST_MIGRATION = '''
from src.shared.infrastructure.database.migration.base import Migration

TABLE = "{table}"


class CreateScratchItems(Migration):
    version = "{version}"
    description = "Create the scratch items table"

    def get_artifacts(self):
        return {{
            "tables": [TABLE],
            "indexes": [(TABLE, "idx_owner_created")],
            "columns": [
                (TABLE, "item_id", "Utf8"),
                (TABLE, "owner", "Utf8"),
                (TABLE, "note", "Utf8?"),
                (TABLE, "payload", "Json?"),
                (TABLE, "amount", "Int32"),
                (TABLE, "created_at", "Timestamp"),
            ],
        }}

    async def up(self, pool):
        await pool.execute_with_retries(
            f"""
            CREATE TABLE IF NOT EXISTS `{{TABLE}}` (
                item_id Utf8 NOT NULL,
                owner Utf8 NOT NULL,
                note Utf8,
                payload Json,
                amount Int32 NOT NULL,
                created_at Timestamp NOT NULL,
                PRIMARY KEY (item_id),
                INDEX idx_owner_created GLOBAL ON (owner, created_at)
            );
            """
        )
'''

_OVERCLAIMING_MIGRATION = '''
from src.shared.infrastructure.database.migration.base import Migration

TABLE = "{table}"


class CreateScratchLabels(Migration):
    version = "{version}"
    description = "Create the scratch labels table and claim more than that"

    def get_artifacts(self):
        return {{
            "tables": [TABLE],
            "indexes": [(TABLE, "idx_never_created")],
            "columns": [(TABLE, "label_id", "Utf8"), (TABLE, "title", "Utf8")],
        }}

    async def up(self, pool):
        await pool.execute_with_retries(
            f"""
            CREATE TABLE IF NOT EXISTS `{{TABLE}}` (
                label_id Utf8 NOT NULL,
                title Utf8,
                PRIMARY KEY (label_id)
            );
            """
        )
'''


@dataclass
class Scratch:
    """Unique names for one test plus the list of what has to be removed."""

    suffix: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tables: List[str] = field(default_factory=list)
    modules: List[str] = field(default_factory=list)

    def table(self, stem: str) -> str:
        name = f"zz_it_{stem}_{self.suffix}"
        self.tables.append(name)
        return name

    def module(self, stem: str) -> str:
        name = f"zz_it_{stem}_{self.suffix}"
        self.modules.append(name)
        return name


@pytest.fixture
async def scratch(ydb_connection: YDBConnection) -> AsyncIterator[Scratch]:
    names = Scratch()
    try:
        yield names
    finally:
        for table in names.tables:
            if await schema.table_exists(
                ydb_connection.driver, ydb_connection.database, table
            ):
                await ydb_connection.pool.execute_with_retries(f"DROP TABLE `{table}`;")
        history_exists = await schema.table_exists(
            ydb_connection.driver, ydb_connection.database, MIGRATIONS_TABLE
        )
        for module in names.modules if history_exists else []:
            await ydb_connection.pool.execute_with_retries(
                f"DECLARE $module AS Utf8; "
                f"DELETE FROM {MIGRATIONS_TABLE} WHERE module_name = $module;",
                parameters={"$module": ydb.TypedValue(module, ydb.PrimitiveType.Utf8)},
            )


def _write_migration(src_root: Path, module: str, version: str, source: str) -> Path:
    versions = src_root / module / "adapters" / "persistence" / "migrations" / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    path = versions / f"{version}_scratch_change.py"
    path.write_text(source, encoding="utf-8")
    return path


async def test_migration_is_applied_verified_and_not_applied_twice(
    ydb_connection: YDBConnection, scratch: Scratch, tmp_path: Path
) -> None:
    src_root = tmp_path / "src"
    module = scratch.module("items")
    table = scratch.table("items")
    version = "20990101000000"
    _write_migration(
        src_root, module, version, _HONEST_MIGRATION.format(table=table, version=version)
    )
    driver, database = ydb_connection.driver, ydb_connection.database

    applied = await apply_pending_migrations(
        ydb_connection, applied_by=APPLIED_BY, src_root=src_root
    )

    assert [(m.module_name, m.version) for m in applied] == [(module, version)]

    record = (await get_applied_migrations(ydb_connection.pool))[f"{module}:{version}"]
    assert record.status == "completed"
    assert record.error_message is None
    assert record.applied_by == APPLIED_BY
    assert record.description == "Create the scratch items table"
    assert record.applied_at.tzinfo is not None

    # The same facts, read straight from the Table Service.
    assert await schema.table_exists(driver, database, table)
    assert await schema.index_exists(driver, database, table, "idx_owner_created")
    assert not await schema.index_exists(driver, database, table, "idx_absent")
    assert await schema.column_exists(driver, database, table, "item_id", "Utf8")
    assert await schema.column_exists(driver, database, table, "note", "Utf8?")
    assert await schema.column_exists(driver, database, table, "payload", "Optional<Json>")
    assert await schema.column_exists(driver, database, table, "amount")
    assert not await schema.column_exists(driver, database, table, "note", "Utf8")
    assert not await schema.column_exists(driver, database, table, "amount", "Int64")
    assert not await schema.column_exists(driver, database, table, "absent")
    assert await schema.get_table_schema(driver, database, f"{table}_absent") is None

    (migration_file,) = discover_migration_files(src_root)
    await verify_migration_artifacts(
        driver, database, migration_file.import_class()(), migration_file
    )

    assert await apply_pending_migrations(
        ydb_connection, applied_by=APPLIED_BY, src_root=src_root
    ) == []
    unchanged = (await get_applied_migrations(ydb_connection.pool))[f"{module}:{version}"]
    assert unchanged.applied_at == record.applied_at


async def test_migration_that_declares_more_than_it_creates_fails(
    ydb_connection: YDBConnection, scratch: Scratch, tmp_path: Path
) -> None:
    src_root = tmp_path / "src"
    module = scratch.module("labels")
    table = scratch.table("labels")
    version = "20990101000100"
    _write_migration(
        src_root, module, version, _OVERCLAIMING_MIGRATION.format(table=table, version=version)
    )

    with pytest.raises(MigrationVerificationError) as raised:
        await apply_pending_migrations(ydb_connection, applied_by=APPLIED_BY, src_root=src_root)

    message = str(raised.value)
    assert "(2 errors)" in message
    assert f"declared index 'idx_never_created' on table '{table}' but it does not exist" in message
    assert "declared column 'title' of type 'Utf8'" in message
    assert "the actual type is 'Utf8?'" in message

    record = (await get_applied_migrations(ydb_connection.pool))[f"{module}:{version}"]
    assert record.status == "failed"
    assert "idx_never_created" in record.error_message

    # A failed migration stays queued, so the next run attempts it again.
    discovered = discover_migration_files(src_root)
    pending = await plan_migrations_to_run(ydb_connection.pool, discovered)
    assert [(m.module_name, m.version) for m in pending] == [(module, version)]


async def test_idempotent_alter_table_helpers(
    ydb_connection: YDBConnection, scratch: Scratch
) -> None:
    table = scratch.table("ops")
    driver, database = ydb_connection.driver, ydb_connection.database
    await ydb_connection.pool.execute_with_retries(
        f"""
        CREATE TABLE `{table}` (
            row_id Utf8 NOT NULL,
            owner Utf8 NOT NULL,
            PRIMARY KEY (row_id)
        );
        """
    )

    for _ in range(2):
        await operations.add_column_if_not_exists(ydb_connection, table, "score", "Int32?")
    assert await schema.column_exists(driver, database, table, "score", "Int32?")

    custom_query = f"ALTER TABLE `{table}` ADD COLUMN `remark` Utf8;"
    for _ in range(2):
        await operations.add_column_if_not_exists(
            ydb_connection, table, "remark", "Utf8?", custom_query
        )
    assert await schema.column_exists(driver, database, table, "remark", "Utf8?")

    with pytest.raises(operations.SchemaMismatchError, match="has type 'Int32\\?'"):
        await operations.add_column_if_not_exists(ydb_connection, table, "score", "Utf8?")

    index_query = f"ALTER TABLE `{table}` ADD INDEX `idx_owner` GLOBAL ON (`owner`);"
    for _ in range(2):
        await operations.add_index_if_not_exists(ydb_connection, table, "idx_owner", index_query)
    assert await schema.index_exists(driver, database, table, "idx_owner")

    # The server itself refuses a second index of the same name; the helper
    # must recognise that answer when it loses a race with another process.
    with pytest.raises(ydb.Error) as duplicate:
        await ydb_connection.pool.execute_with_retries(index_query)
    assert operations.reports_existing_object(duplicate.value)

    for _ in range(2):
        await operations.drop_column_if_exists(ydb_connection, table, "score")
    assert not await schema.column_exists(driver, database, table, "score")
    assert await schema.column_exists(driver, database, table, "remark")

    with pytest.raises(LookupError, match="does not exist"):
        await operations.drop_column_if_exists(ydb_connection, f"{table}_absent", "score")


async def test_project_migrations_create_every_declared_artifact(
    ydb_connection: YDBConnection,
) -> None:
    discovered = discover_migration_files()
    if not discovered:
        pytest.skip("the project tree holds no migration files")
    driver, database = ydb_connection.driver, ydb_connection.database

    await apply_pending_migrations(ydb_connection, applied_by=APPLIED_BY)

    applied = await get_applied_migrations(ydb_connection.pool)
    for migration_file in discovered:
        key = f"{migration_file.module_name}:{migration_file.version}"
        assert applied[key].status == "completed", key

        migration = migration_file.import_class()()
        artifacts = migration.get_artifacts()
        assert artifacts.get("tables"), f"{key} declares no tables"
        await verify_migration_artifacts(driver, database, migration, migration_file)
        for table in artifacts["tables"]:
            assert await schema.table_exists(driver, database, table), (key, table)

    assert await apply_pending_migrations(ydb_connection, applied_by=APPLIED_BY) == []
