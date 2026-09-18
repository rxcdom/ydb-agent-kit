"""Checks what migrations declare against the live schema and the history."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import ydb

from src.shared.infrastructure.database.ydb.types import normalize_type_name

from .base import Migration
from .discovery import MigrationFile
from .repository import (
    STATUS_FAILED,
    STATUS_IN_PROGRESS,
    get_applied_migrations,
    migration_key,
)
from . import schema as db_schema

TableSchemas = Dict[str, Optional[ydb.TableSchemeEntry]]


class MigrationVerificationError(RuntimeError):
    """Declared artifacts are missing, or a migration has no terminal status."""


def _numbered(errors: List[str]) -> str:
    return "\n".join(f"  {number}. {error}" for number, error in enumerate(errors, 1))


def _plural(count: int) -> str:
    return "error" if count == 1 else "errors"


def _declared_table_names(artifacts: Dict[str, Any]) -> Set[str]:
    """Collect every table a declaration refers to, ignoring malformed entries.

    Malformed entries are reported by the per-kind checks; here they are only
    kept out of the schema lookups.
    """
    names: Set[str] = set()
    tables = artifacts.get("tables", [])
    if isinstance(tables, list):
        names.update(name for name in tables if isinstance(name, str))
    for kind in ("indexes", "columns"):
        entries = artifacts.get(kind, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str):
                names.add(entry[0])
    return names


def _check_tables(label: str, tables: Any, schemas: TableSchemas) -> List[str]:
    if not isinstance(tables, list):
        return [
            f"{label} has an invalid 'tables' declaration: expected a list, "
            f"got {type(tables).__name__}"
        ]
    errors: List[str] = []
    for table_name in tables:
        if not isinstance(table_name, str):
            errors.append(
                f"{label} has an invalid table name: expected a string, "
                f"got {type(table_name).__name__}"
            )
        elif schemas.get(table_name) is None:
            errors.append(f"{label} declared table '{table_name}' but it does not exist")
    return errors


def _check_indexes(label: str, indexes: Any, schemas: TableSchemas) -> List[str]:
    if not isinstance(indexes, list):
        return [
            f"{label} has an invalid 'indexes' declaration: expected a list, "
            f"got {type(indexes).__name__}"
        ]
    errors: List[str] = []
    for entry in indexes:
        if (
            not isinstance(entry, (tuple, list))
            or len(entry) != 2
            or not all(isinstance(part, str) for part in entry)
        ):
            errors.append(
                f"{label} has an invalid index declaration: {entry!r}. "
                f"Expected a (table_name, index_name) pair of strings"
            )
            continue

        table_name, index_name = entry
        table = schemas.get(table_name)
        if table is None:
            errors.append(
                f"{label} declared index '{index_name}' on table '{table_name}' "
                f"but the table does not exist"
            )
        elif db_schema.find_index(table, index_name) is None:
            errors.append(
                f"{label} declared index '{index_name}' on table '{table_name}' "
                f"but it does not exist"
            )
    return errors


def _check_columns(label: str, columns: Any, schemas: TableSchemas) -> List[str]:
    if not isinstance(columns, list):
        return [
            f"{label} has an invalid 'columns' declaration: expected a list, "
            f"got {type(columns).__name__}"
        ]
    errors: List[str] = []
    for entry in columns:
        if (
            not isinstance(entry, (tuple, list))
            or len(entry) not in (2, 3)
            or not all(isinstance(part, str) for part in entry)
        ):
            errors.append(
                f"{label} has an invalid column declaration: {entry!r}. "
                f"Expected (table, column) or (table, column, type) strings"
            )
            continue

        table_name, column_name = entry[0], entry[1]
        expected_type = entry[2] if len(entry) == 3 else None

        table = schemas.get(table_name)
        if table is None:
            errors.append(
                f"{label} declared column '{column_name}' on table '{table_name}' "
                f"but the table does not exist"
            )
            continue

        column = db_schema.find_column(table, column_name)
        if column is None:
            errors.append(
                f"{label} declared column '{column_name}' on table '{table_name}' "
                f"but it does not exist"
            )
            continue
        if expected_type is None:
            continue

        try:
            matches = db_schema.column_type_matches(column, expected_type)
            actual_type = db_schema.get_column_type_string(column)
        except TypeError as error:
            errors.append(
                f"{label} declared column '{column_name}' of type '{expected_type}' on table "
                f"'{table_name}' but its actual type cannot be rendered: {error}"
            )
            continue
        if not matches:
            errors.append(
                f"{label} declared column '{column_name}' of type '{expected_type}' on table "
                f"'{table_name}' but the actual type is '{actual_type}'"
                + _nullability_hint(expected_type, actual_type)
            )
    return errors


def _nullability_hint(expected_type: str, actual_type: str) -> str:
    """Explain the most common mismatch: a column that was not created ``NOT NULL``."""
    if normalize_type_name(actual_type) == f"{normalize_type_name(expected_type)}?":
        return " (a column is nullable unless it is created NOT NULL)"
    return ""


async def verify_migration_artifacts(
    driver: ydb.aio.Driver,
    database: str,
    migration_instance: Migration,
    migration_file: MigrationFile,
) -> None:
    """Check that everything a migration declares exists in the live schema.

    Each referenced table is described once. Every problem is collected, so a
    single failure report lists all missing tables, indexes and columns and all
    type mismatches.

    Raises:
        MigrationVerificationError: at least one declared artifact is missing,
            mistyped or malformed.
    """
    artifacts = migration_instance.get_artifacts()
    if not artifacts:
        return

    label = f"Migration {migration_key(migration_file.module_name, migration_file.version)}"
    if not isinstance(artifacts, dict):
        raise MigrationVerificationError(
            f"{label}: get_artifacts() must return a dictionary, "
            f"got {type(artifacts).__name__}"
        )

    errors: List[str] = []
    schemas: TableSchemas = {}
    for table_name in sorted(_declared_table_names(artifacts)):
        try:
            schemas[table_name] = await db_schema.get_table_schema(driver, database, table_name)
        except (ValueError, ydb.Error) as error:
            errors.append(f"{label}: cannot read the schema of table '{table_name}': {error}")
            schemas[table_name] = None

    errors.extend(_check_tables(label, artifacts.get("tables", []), schemas))
    errors.extend(_check_indexes(label, artifacts.get("indexes", []), schemas))
    errors.extend(_check_columns(label, artifacts.get("columns", []), schemas))

    if errors:
        raise MigrationVerificationError(
            f"Migration verification failed ({len(errors)} {_plural(len(errors))}):\n"
            + _numbered(errors)
        )


async def verify_all_migrations_completed(
    pool: ydb.aio.QuerySessionPool, discovered_files: List[MigrationFile]
) -> None:
    """Refuse to continue while a discovered migration is failed or unfinished.

    Raises:
        MigrationVerificationError: a migration is recorded as ``failed`` or is
            still ``in_progress``.
    """
    applied = await get_applied_migrations(pool)
    errors: List[str] = []

    for migration_file in discovered_files:
        key = migration_key(migration_file.module_name, migration_file.version)
        record = applied.get(key)
        if record is None:
            continue
        if record.status == STATUS_FAILED:
            errors.append(
                f"Migration {key} has status 'failed'. "
                f"Error: {record.error_message or 'unknown error'}"
            )
        elif record.status == STATUS_IN_PROGRESS:
            errors.append(
                f"Migration {key} has status 'in_progress': a previous run was interrupted."
            )

    if errors:
        raise MigrationVerificationError(
            f"Some migrations have not completed ({len(errors)} {_plural(len(errors))}):\n"
            + _numbered(errors)
        )
