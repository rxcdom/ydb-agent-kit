"""Helpers shared by every YDB data mapper."""
import json
from datetime import datetime, timezone
from typing import Any, List, Optional


def collect_result_set_rows(result_sets: Any) -> List[Any]:
    """Return every row of the first logical result set.

    The Query Service streams a ``SELECT`` back as a sequence of result-set
    *parts*; the server starts a new part roughly every thousand rows or on a
    byte threshold. ``execute_with_retries`` materialises the parts into a flat
    list, so one large result arrives as several entries sharing one ``index``.

    Reading only the first entry would silently drop every row past the first
    part. This helper concatenates the rows of all parts that share the first
    part's ``index``. For a multi-statement query it stops at the second logical
    result set, so the caller still receives exactly the first one, complete.
    """
    if not result_sets:
        return []

    first_index = getattr(result_sets[0], "index", None)
    rows: List[Any] = []
    for result_set in result_sets:
        if getattr(result_set, "index", first_index) != first_index:
            break
        part_rows = getattr(result_set, "rows", None)
        if part_rows:
            rows.extend(part_rows)
    return rows


def read_row_value(row: Any, column: str) -> Any:
    """Read one column of a result row.

    SDK rows expose columns as attributes and as mapping keys; plain dicts are
    accepted as well so that mappers can be exercised without the SDK.
    """
    if isinstance(row, dict):
        if column not in row:
            raise ValueError(f"Missing column '{column}' in the result row.")
        return row[column]

    if hasattr(row, column):
        return getattr(row, column)

    try:
        return row[column]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            f"Missing column '{column}' in a result row of type {type(row).__name__}."
        ) from error


def normalize_datetime_to_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Return a timezone-aware UTC datetime.

    YDB hands back naive datetimes for ``Timestamp`` columns; they are UTC by
    definition. Aware values are converted.
    """
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc)
    return value.replace(tzinfo=timezone.utc)


def safe_decode(value: Any, field_name: str) -> str:
    """Return a required text column as ``str``.

    ``Utf8`` columns arrive as ``str`` and ``String`` columns as ``bytes``; both
    are accepted so a mapper does not depend on the column flavour.
    """
    if value is None:
        raise ValueError(f"Required field '{field_name}' is None")
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Invalid UTF-8 in field '{field_name}': {error}") from error
    return str(value)


def optional_decode(value: Any, field_name: str) -> Optional[str]:
    """``safe_decode`` for a nullable text column."""
    if value is None:
        return None
    return safe_decode(value, field_name)


def safe_parse_json(value: Any, field_name: str) -> Any:
    """Parse a ``Json`` column.

    Depending on the SDK path the value arrives already parsed, as a JSON
    string, or as UTF-8 bytes of one.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Failed to decode bytes in field '{field_name}': {error}") from error

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in field '{field_name}': {error}. Value preview: {value[:200]}"
            ) from error

    raise ValueError(
        f"Unsupported value of type {type(value).__name__} in JSON field '{field_name}'"
    )
