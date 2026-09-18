"""The single table of YQL type names.

Two consumers need to turn an SDK type object into its YQL spelling: the
repository when it renders ``DECLARE`` statements, and the migration framework
when it compares a declared column type with the live schema.
"""
from typing import Any

import ydb

_PRIMITIVE_TYPE_NAMES = {
    ydb.PrimitiveType.Bool: "Bool",
    ydb.PrimitiveType.Int8: "Int8",
    ydb.PrimitiveType.Uint8: "Uint8",
    ydb.PrimitiveType.Int16: "Int16",
    ydb.PrimitiveType.Uint16: "Uint16",
    ydb.PrimitiveType.Int32: "Int32",
    ydb.PrimitiveType.Uint32: "Uint32",
    ydb.PrimitiveType.Int64: "Int64",
    ydb.PrimitiveType.Uint64: "Uint64",
    ydb.PrimitiveType.Float: "Float",
    ydb.PrimitiveType.Double: "Double",
    ydb.PrimitiveType.String: "String",
    ydb.PrimitiveType.Utf8: "Utf8",
    ydb.PrimitiveType.Json: "Json",
    ydb.PrimitiveType.JsonDocument: "JsonDocument",
    ydb.PrimitiveType.Yson: "Yson",
    ydb.PrimitiveType.Date: "Date",
    ydb.PrimitiveType.Datetime: "Datetime",
    ydb.PrimitiveType.Timestamp: "Timestamp",
    ydb.PrimitiveType.Interval: "Interval",
}


def primitive_type_name(primitive_type: Any) -> str:
    """Return the YQL name of a primitive type; unknown types are an error."""
    try:
        return _PRIMITIVE_TYPE_NAMES[primitive_type]
    except (KeyError, TypeError) as error:
        raise TypeError(f"Unsupported YDB primitive type: {primitive_type!r}") from error


def format_ydb_type(ydb_type: Any) -> str:
    """Render a type for a ``DECLARE`` statement: ``Utf8`` or ``Timestamp?``."""
    if isinstance(ydb_type, ydb.OptionalType):
        return f"{primitive_type_name(ydb_type.item)}?"
    return primitive_type_name(ydb_type)


def normalize_type_name(type_name: str) -> str:
    """Bring a type spelling to the canonical ``Name`` / ``Name?`` form.

    ``Optional<Utf8>`` and ``Utf8?`` describe the same column, and so do
    spellings that differ only in case or surrounding whitespace.
    """
    candidate = type_name.strip()
    optional = False

    if candidate.endswith("?"):
        optional = True
        candidate = candidate[:-1].strip()
    lowered = candidate.lower()
    if lowered.startswith("optional<") and lowered.endswith(">"):
        optional = True
        candidate = candidate[len("optional<"):-1].strip()

    canonical_names = {name.lower(): name for name in _PRIMITIVE_TYPE_NAMES.values()}
    canonical = canonical_names.get(candidate.lower(), candidate)
    return f"{canonical}?" if optional else canonical
