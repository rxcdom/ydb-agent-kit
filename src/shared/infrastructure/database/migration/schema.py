"""Read-only inspection of the live schema.

Everything here goes through the Table Service ``describe_table`` call, the
only API that reports columns with their types and secondary indexes. The
module performs no writes and depends on no other part of the framework except
its console log, so both ``operations`` and ``verification`` can build on it.

The Table Service addresses a table by its absolute path, which is why every
function takes the database path next to the driver.
"""
from __future__ import annotations

import re
from typing import Optional

import ydb

from src.shared.infrastructure.database.ydb.types import format_ydb_type, normalize_type_name

from .logging import log_warning

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_NOT_FOUND_PHRASES = (
    "no such table",
    "path does not exist",
    "does not exist",
    "cannot find table",
)
_NOT_FOUND_PATTERNS = (
    re.compile(r"table.*not found"),
    re.compile(r"column.*not found"),
    re.compile(r"index.*not found"),
)


def is_not_found_error(error: ydb.Error) -> bool:
    """Tell whether a Query Service error reports a missing table, column or index.

    The Query Service signals a missing object through the issue text only, so
    the text is matched in one place instead of at every call site.
    """
    message = str(error).lower()
    if any(phrase in message for phrase in _NOT_FOUND_PHRASES):
        return True
    return any(pattern.search(message) for pattern in _NOT_FOUND_PATTERNS)


def validate_identifier(identifier: str, identifier_type: str) -> None:
    """Reject a table, column or index name that is unsafe to splice into YQL."""
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"Invalid {identifier_type} name: must be a non-empty string")
    if not _IDENTIFIER_PATTERN.match(identifier):
        raise ValueError(
            f"Invalid {identifier_type} name {identifier!r}: must start with a letter or an "
            f"underscore and contain only letters, digits and underscores"
        )


def table_path(database: str, table_name: str) -> str:
    """Build the absolute path the Table Service expects."""
    return f"{database.rstrip('/')}/{table_name}"


async def get_table_schema(
    driver: ydb.aio.Driver, database: str, table_name: str
) -> Optional[ydb.TableSchemeEntry]:
    """Describe a table; ``None`` when it does not exist.

    A Table Service session has to be created on the server before it can
    serve a request, and it is deleted afterwards so the inspection leaves no
    session behind. ``describe_table`` answers with a scheme error when the
    path is absent or is not a table; every other error propagates.
    """
    validate_identifier(table_name, "table")

    session = await driver.table_client.session().create()
    try:
        return await session.describe_table(table_path(database, table_name))
    except ydb.issues.SchemeError:
        return None
    finally:
        try:
            await session.delete()
        except ydb.Error as error:
            log_warning(
                f"Could not delete the schema inspection session: "
                f"{type(error).__name__}: {error}",
                indent=2,
            )


def find_index(table: ydb.TableSchemeEntry, index_name: str) -> Optional[ydb.TableIndex]:
    for index in table.indexes or []:
        if index.name == index_name:
            return index
    return None


def find_column(table: ydb.TableSchemeEntry, column_name: str) -> Optional[ydb.Column]:
    for column in table.columns or []:
        if column.name == column_name:
            return column
    return None


def get_column_type_string(column: ydb.Column) -> str:
    """Return the YQL spelling of a described column type, e.g. ``Timestamp?``."""
    return format_ydb_type(column.type)


def column_type_matches(column: ydb.Column, expected_type: str) -> bool:
    """Compare a described column with a declared type spelling.

    Optionality is part of the type: a nullable column is ``Utf8?`` and matches
    neither ``Utf8`` nor the other way round.
    """
    actual = normalize_type_name(get_column_type_string(column))
    return actual == normalize_type_name(expected_type)


async def table_exists(driver: ydb.aio.Driver, database: str, table_name: str) -> bool:
    return await get_table_schema(driver, database, table_name) is not None


async def index_exists(
    driver: ydb.aio.Driver, database: str, table_name: str, index_name: str
) -> bool:
    validate_identifier(index_name, "index")
    table = await get_table_schema(driver, database, table_name)
    return table is not None and find_index(table, index_name) is not None


async def column_exists(
    driver: ydb.aio.Driver,
    database: str,
    table_name: str,
    column_name: str,
    expected_type: Optional[str] = None,
) -> bool:
    """Tell whether a column exists and, when a type is given, has that type."""
    validate_identifier(column_name, "column")
    table = await get_table_schema(driver, database, table_name)
    if table is None:
        return False
    column = find_column(table, column_name)
    if column is None:
        return False
    if expected_type is None:
        return True
    return column_type_matches(column, expected_type)
