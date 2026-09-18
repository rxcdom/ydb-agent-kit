"""Idempotent ``ALTER TABLE`` helpers.

Each helper inspects the live schema first and changes it only when needed, so
calling it again is a no-op. Schema inspection goes through the Table Service
and the change itself through the Query Service, which is why the helpers take
the whole connection: the pool, the driver and the database path.

Dependencies: ``schema`` for inspection and ``logging`` for output; nothing in
``schema`` depends on this module.
"""
from __future__ import annotations

from typing import Optional

import ydb

from src.shared.infrastructure.database.ydb.connection import YDBConnection

from . import schema as db_schema
from .logging import log_info, log_success

_ALREADY_EXISTS_PHRASES = ("already exists", "path exist")


class SchemaMismatchError(RuntimeError):
    """The live schema contradicts what the caller expects to be there."""


def reports_existing_object(error: ydb.Error) -> bool:
    message = str(error).lower()
    return any(phrase in message for phrase in _ALREADY_EXISTS_PHRASES)


async def add_index_if_not_exists(
    connection: YDBConnection, table_name: str, index_name: str, index_query: str
) -> None:
    """Run ``index_query`` unless the table already has the index.

    ``index_query`` is the full statement, for example
    ``ALTER TABLE tasks ADD INDEX idx_tasks_user_due GLOBAL ON (user_id, due_at);``.
    Another process may create the index between the check and the statement;
    the server then reports an existing path, which is the desired end state.
    """
    if await db_schema.index_exists(
        connection.driver, connection.database, table_name, index_name
    ):
        log_info(f"Index '{index_name}' already exists on table '{table_name}', skipping.")
        return

    try:
        await connection.pool.execute_with_retries(index_query)
    except ydb.Error as error:
        if not reports_existing_object(error):
            raise
        log_info(f"Index '{index_name}' on table '{table_name}' was created concurrently.")
        return
    log_success(f"Created index '{index_name}' on table '{table_name}'.")


async def add_column_if_not_exists(
    connection: YDBConnection,
    table_name: str,
    column_name: str,
    column_type: str,
    column_query: Optional[str] = None,
) -> None:
    """Add a column unless the table already has it with the same type.

    ``column_type`` is the type the Table Service reports once the column
    exists. YDB adds columns as nullable, so the usual spelling is ``Int32?``
    or ``Utf8?``. Without ``column_query`` the statement
    ``ALTER TABLE <table> ADD COLUMN <column> <column_type>;`` is generated.

    Raises:
        SchemaMismatchError: the column exists, or ends up existing, with a
            type other than ``column_type``.
    """
    db_schema.validate_identifier(table_name, "table")
    db_schema.validate_identifier(column_name, "column")

    if await _has_column_of_type(connection, table_name, column_name, column_type):
        log_info(f"Column '{column_name}' already exists on table '{table_name}', skipping.")
        return

    query = column_query or (
        f"ALTER TABLE `{table_name}` ADD COLUMN `{column_name}` {column_type};"
    )
    try:
        await connection.pool.execute_with_retries(query)
    except ydb.Error as error:
        if not reports_existing_object(error):
            raise
        log_info(f"Column '{column_name}' on table '{table_name}' was created concurrently.")
    else:
        log_success(f"Created column '{column_name}' on table '{table_name}'.")

    if not await _has_column_of_type(connection, table_name, column_name, column_type):
        raise SchemaMismatchError(
            f"Column '{column_name}' was not found on table '{table_name}' after it was added"
        )


async def _has_column_of_type(
    connection: YDBConnection, table_name: str, column_name: str, column_type: str
) -> bool:
    """``True`` for a matching column, ``False`` for an absent one.

    A column that exists with another type is an error: adding it again could
    never produce the requested schema.
    """
    table = await db_schema.get_table_schema(
        connection.driver, connection.database, table_name
    )
    if table is None:
        return False
    column = db_schema.find_column(table, column_name)
    if column is None:
        return False
    if not db_schema.column_type_matches(column, column_type):
        raise SchemaMismatchError(
            f"Column '{column_name}' on table '{table_name}' has type "
            f"'{db_schema.get_column_type_string(column)}', expected '{column_type}'"
        )
    return True


async def drop_column_if_exists(
    connection: YDBConnection, table_name: str, column_name: str
) -> None:
    """Drop a column unless it is already gone.

    Raises:
        LookupError: the table itself does not exist, which points at a
            mistake in the caller rather than at an already applied change.
    """
    db_schema.validate_identifier(table_name, "table")
    db_schema.validate_identifier(column_name, "column")

    table = await db_schema.get_table_schema(
        connection.driver, connection.database, table_name
    )
    if table is None:
        raise LookupError(
            f"Cannot drop column '{column_name}': table '{table_name}' does not exist"
        )
    if db_schema.find_column(table, column_name) is None:
        log_info(f"Column '{column_name}' does not exist on table '{table_name}', skipping.")
        return

    await connection.pool.execute_with_retries(
        f"ALTER TABLE `{table_name}` DROP COLUMN `{column_name}`;"
    )
    log_success(f"Dropped column '{column_name}' from table '{table_name}'.")
