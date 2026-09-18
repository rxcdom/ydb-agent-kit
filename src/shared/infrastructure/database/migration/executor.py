"""Runs one migration and records what happened to it."""
from __future__ import annotations

import traceback

import ydb

from src.shared.infrastructure.database.ydb.connection import YDBConnection

from .base import Migration
from .discovery import MigrationFile, compute_checksum
from .logging import (
    log_error,
    log_info,
    log_migration_error,
    log_migration_start,
    log_migration_success,
    log_step,
    log_success,
)
from .repository import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_IN_PROGRESS,
    record_migration,
)
from .verification import verify_migration_artifacts


async def run_migration(
    connection: YDBConnection, migration_file: MigrationFile, applied_by: str
) -> None:
    """Apply a single migration.

    Lifecycle:
    1. the history row is written as ``in_progress``;
    2. ``up()`` runs;
    3. declared artifacts are verified against the live schema;
    4. the row becomes ``completed``.

    Any error in steps 2 and 3 turns the row into ``failed`` with the error
    text, and the error propagates so the caller stops the whole run. A failed
    verification is a failed migration: the declared schema is the contract.
    """
    migration_cls = migration_file.import_class()
    migration_instance: Migration = migration_cls()

    version = migration_file.version
    module_name = migration_file.module_name
    description = getattr(migration_cls, "description", "No description")
    checksum = compute_checksum(migration_file)

    async def record(status: str, error_message: str | None = None) -> None:
        await record_migration(
            connection.pool,
            version=version,
            module_name=module_name,
            description=description,
            checksum=checksum,
            applied_by=applied_by,
            status=status,
            error_message=error_message,
        )

    log_migration_start(module_name, version, description)

    log_step(1, 3, "Recording migration status")
    await record(STATUS_IN_PROGRESS)

    try:
        log_step(2, 3, "Executing migration")
        await migration_instance.up(connection.pool)
        log_success("Migration execution completed", indent=2)

        if migration_instance.get_artifacts():
            log_step(3, 3, "Verifying migration artifacts")
            await verify_migration_artifacts(
                connection.driver, connection.database, migration_instance, migration_file
            )
            log_success("All artifacts verified", indent=2)
        else:
            log_info("No artifacts declared, skipping verification", indent=2)
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        log_migration_error(module_name, version, error_message)
        log_error(f"Traceback:\n{traceback.format_exc()}", indent=2)
        try:
            await record(STATUS_FAILED, error_message)
        except ydb.Error as record_error:
            # The original failure is what the operator needs to see; the row
            # stays "in_progress" and the planner queues the migration again.
            log_error(f"Could not record the failed status: {record_error}", indent=2)
        raise

    await record(STATUS_COMPLETED)
    log_migration_success(module_name, version)
