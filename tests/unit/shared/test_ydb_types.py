import pytest
import ydb

from src.shared.infrastructure.database.ydb.types import (
    format_ydb_type,
    normalize_type_name,
    primitive_type_name,
)


def test_primitive_and_optional_types_render_for_declare():
    assert format_ydb_type(ydb.PrimitiveType.Utf8) == "Utf8"
    assert format_ydb_type(ydb.OptionalType(ydb.PrimitiveType.Timestamp)) == "Timestamp?"
    assert primitive_type_name(ydb.PrimitiveType.Json) == "Json"


def test_unknown_type_is_an_error_not_a_guess():
    with pytest.raises(TypeError, match="Unsupported YDB primitive type"):
        primitive_type_name("not-a-type")


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("Utf8", "Utf8"),
        (" utf8 ", "Utf8"),
        ("Utf8?", "Utf8?"),
        ("Optional<Utf8>", "Utf8?"),
        ("optional<timestamp>", "Timestamp?"),
        ("Int32 ?", "Int32?"),
    ],
)
def test_type_spellings_normalise_to_one_form(spelling, expected):
    assert normalize_type_name(spelling) == expected
