"""Tests of the shared YDB helpers, centred on multi-part result-set assembly."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import pytest

from src.shared.infrastructure.database.ydb.utils import (
    collect_result_set_rows,
    normalize_datetime_to_utc,
    optional_decode,
    read_row_value,
    safe_decode,
    safe_parse_json,
)


@dataclass
class _FakeResultSet:
    rows: List[Any]
    index: Optional[int] = 0


def test_empty_or_none_inputs_return_empty_list():
    assert collect_result_set_rows(None) == []
    assert collect_result_set_rows([]) == []
    assert collect_result_set_rows([_FakeResultSet(rows=[])]) == []


def test_single_part_returns_its_rows():
    assert collect_result_set_rows([_FakeResultSet(rows=[1, 2, 3])]) == [1, 2, 3]


def test_multi_part_single_select_concatenates_every_part():
    """A result larger than one part must come back whole, not cut at part 0."""
    part0 = _FakeResultSet(rows=list(range(0, 1000)), index=0)
    part1 = _FakeResultSet(rows=list(range(1000, 2000)), index=0)
    part2 = _FakeResultSet(rows=list(range(2000, 2345)), index=0)

    rows = collect_result_set_rows([part0, part1, part2])

    assert rows == list(range(0, 2345))


def test_stops_at_second_logical_result_set():
    first_a = _FakeResultSet(rows=["a1", "a2"], index=0)
    first_b = _FakeResultSet(rows=["a3"], index=0)
    second = _FakeResultSet(rows=["b1", "b2"], index=1)

    assert collect_result_set_rows([first_a, first_b, second]) == ["a1", "a2", "a3"]


def test_missing_index_attribute_is_treated_as_one_result_set():
    class _Bare:
        def __init__(self, rows):
            self.rows = rows

    assert collect_result_set_rows([_Bare(["x", "y"]), _Bare(["z"])]) == ["x", "y", "z"]


def test_read_row_value_supports_attributes_mappings_and_reports_missing_columns():
    class _Row:
        title = "from attribute"

    assert read_row_value(_Row(), "title") == "from attribute"
    assert read_row_value({"title": "from dict"}, "title") == "from dict"
    with pytest.raises(ValueError, match="Missing column 'absent'"):
        read_row_value({"title": "x"}, "absent")
    with pytest.raises(ValueError, match="Missing column 'absent'"):
        read_row_value(_Row(), "absent")


def test_naive_datetime_is_taken_as_utc_and_aware_datetime_is_converted():
    naive = datetime(2026, 9, 1, 12, 0)
    aware = datetime(2026, 9, 1, 14, 0, tzinfo=timezone(timedelta(hours=2)))

    assert normalize_datetime_to_utc(naive) == datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    assert normalize_datetime_to_utc(aware) == datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    assert normalize_datetime_to_utc(None) is None


def test_decode_accepts_text_and_bytes_and_rejects_null_for_required_fields():
    assert safe_decode("text", "f") == "text"
    assert safe_decode(b"bytes", "f") == "bytes"
    assert optional_decode(None, "f") is None
    with pytest.raises(ValueError, match="Required field 'f' is None"):
        safe_decode(None, "f")
    with pytest.raises(ValueError, match="Invalid UTF-8"):
        safe_decode(b"\xff\xfe", "f")


def test_json_is_parsed_from_every_transport_form():
    assert safe_parse_json(None, "f") is None
    assert safe_parse_json({"a": 1}, "f") == {"a": 1}
    assert safe_parse_json('{"a": [1, 2]}', "f") == {"a": [1, 2]}
    assert safe_parse_json(b'[{"a": 1}]', "f") == [{"a": 1}]
    with pytest.raises(ValueError, match="Invalid JSON in field 'f'"):
        safe_parse_json("{'python': 'repr'}", "f")
    with pytest.raises(ValueError, match="Unsupported value"):
        safe_parse_json(42, "f")
