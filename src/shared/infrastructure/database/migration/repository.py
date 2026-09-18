"""Reads and writes the migration history table."""
from __future__ import annotations

import datetime as dt
from typing import Dict, Optional

import ydb

from src.shared.domain.migration import MigrationRecord
from src.shared.infrastructure.database.ydb.utils import (
    collect_result_set_rows,
    normalize_datetime_to_utc,
    optional_decode,
    read_row_value,
    safe_decode,
)

from .logging import log_error, log_info, log_success
from .schema import is_not_found_error

MIGRATIONS_TABLE = "schema_migrations"

STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

MAX_ERROR_MESSAGE_LENGTH = 1000


def migration_key(module_name: str, version: str) -> str:
    """Key of a migration in the applied-history mapping."""
    return f"{module_name}:{version}"


async def ensure_migrations_table(pool: ydb.aio.QuerySessionPool) -> None:
    """Create the history table if it does not exist.

    This is the only place the layout of the table is defined.
    """
    log_info(f"Creating table '{MIGRATIONS_TABLE}' if it does not exist...", indent=2)
    try:
        await pool.execute_with_retries(
            f"""
            CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
                module_name Utf8 NOT NULL,
                version Utf8 NOT NULL,
                description Utf8 NOT NULL,
                applied_at Timestamp NOT NULL,
                checksum Utf8 NOT NULL,
                applied_by Utf8 NOT NULL,
                status Utf8 NOT NULL,
                error_message Utf8?,
                PRIMARY KEY (module_name, version)
            );
            """
        )
    except ydb.Error as error:
        log_error(f"Could not create table '{MIGRATIONS_TABLE}': {error}", indent=2)
        raise
    log_success(f"Table '{MIGRATIONS_TABLE}' is ready", indent=2)


async def get_applied_migrations(pool: ydb.aio.QuerySessionPool) -> Dict[str, MigrationRecord]:
    """Return every recorded migration keyed by ``module:version``.

    Reading never changes the schema: when the history table has not been
    created yet the history is simply empty.
    """
    query = f"""
        SELECT module_name, version, description, applied_at,
               checksum, applied_by, status, error_message
        FROM {MIGRATIONS_TABLE};
    """
    try:
        result_sets = await pool.execute_with_retries(query)
    except ydb.issues.SchemeError as error:
        if is_not_found_error(error):
            return {}
        raise

    applied: Dict[str, MigrationRecord] = {}
    for row in collect_result_set_rows(result_sets):
        record = MigrationRecord(
            version=safe_decode(read_row_value(row, "version"), "version"),
            module_name=safe_decode(read_row_value(row, "module_name"), "module_name"),
            description=safe_decode(read_row_value(row, "description"), "description"),
            applied_at=normalize_datetime_to_utc(read_row_value(row, "applied_at")),
            checksum=safe_decode(read_row_value(row, "checksum"), "checksum"),
            applied_by=safe_decode(read_row_value(row, "applied_by"), "applied_by"),
            status=safe_decode(read_row_value(row, "status"), "status"),
            error_message=optional_decode(read_row_value(row, "error_message"), "error_message"),
        )
        applied[migration_key(record.module_name, record.version)] = record
    return applied


def _text(value: str) -> ydb.TypedValue:
    return ydb.TypedValue(value, ydb.PrimitiveType.Utf8)


async def record_migration(
    pool: ydb.aio.QuerySessionPool,
    *,
    version: str,
    module_name: str,
    description: str,
    checksum: str,
    applied_by: str,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    """Upsert the history row of one migration.

    One row exists per ``(module_name, version)``, so each status change
    overwrites the previous one. A long error text is cut to a bounded length.
    """
    query = f"""
        DECLARE $module_name AS Utf8;
        DECLARE $version AS Utf8;
        DECLARE $description AS Utf8;
        DECLARE $applied_at AS Timestamp;
        DECLARE $checksum AS Utf8;
        DECLARE $applied_by AS Utf8;
        DECLARE $status AS Utf8;
        DECLARE $error_message AS Utf8?;

        UPSERT INTO {MIGRATIONS_TABLE} (
            module_name, version, description, applied_at,
            checksum, applied_by, status, error_message
        )
        VALUES (
            $module_name, $version, $description, $applied_at,
            $checksum, $applied_by, $status, $error_message
        );
    """
    stored_error = error_message[:MAX_ERROR_MESSAGE_LENGTH] if error_message else None
    parameters = {
        "$module_name": _text(module_name),
        "$version": _text(version),
        "$description": _text(description),
        "$applied_at": ydb.TypedValue(
            dt.datetime.now(dt.timezone.utc), ydb.PrimitiveType.Timestamp
        ),
        "$checksum": _text(checksum),
        "$applied_by": _text(applied_by),
        "$status": _text(status),
        "$error_message": ydb.TypedValue(
            stored_error, ydb.OptionalType(ydb.PrimitiveType.Utf8)
        ),
    }
    await pool.execute_with_retries(query, parameters=parameters)
