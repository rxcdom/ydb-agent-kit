from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest
import ydb

from src.shared.domain.exceptions import PersistenceError
from src.shared.infrastructure.database.ydb.repository import DataMapper, YDBRepository


@dataclass
class _Note:
    note_id: str
    body: Optional[str]
    payload: Optional[dict]


class _NoteMapper(DataMapper[_Note]):
    def to_domain(self, row: Any) -> _Note:
        return _Note(note_id=row["note_id"], body=row["body"], payload=row["payload"])

    def to_ydb_params(self, entity: _Note) -> Dict[str, Any]:
        return {"$note_id": entity.note_id, "$body": entity.body, "$payload": entity.payload}

    def get_ydb_type_map(self) -> Dict[str, Any]:
        return {
            "$note_id": ydb.PrimitiveType.Utf8,
            "$body": ydb.OptionalType(ydb.PrimitiveType.Utf8),
            "$payload": ydb.OptionalType(ydb.PrimitiveType.Json),
            "$raw": ydb.PrimitiveType.String,
            "$limit": ydb.PrimitiveType.Uint64,
        }

    def get_column_params(self):
        return ["$note_id", "$body", "$payload"]


class _ResultSet:
    def __init__(self, rows):
        self.rows = rows
        self.index = 0


class _FakePool:
    def __init__(self, rows=None, error: Optional[Exception] = None):
        self._rows = rows or []
        self._error = error
        self.calls = []

    async def execute_with_retries(self, query, parameters=None):
        self.calls.append((query, parameters))
        if self._error:
            raise self._error
        return [_ResultSet(self._rows)]


class _FakeTx:
    def __init__(self, parts):
        self._parts = parts
        self.calls = []

    async def execute(self, query, parameters=None, commit_tx=False):
        self.calls.append((query, parameters, commit_tx))

        async def stream():
            for part in self._parts:
                yield part

        return stream()


def _repository(pool) -> YDBRepository:
    return YDBRepository(pool, _NoteMapper(), "notes")


def test_utf8_passes_through_and_string_is_encoded():
    params = _repository(_FakePool())._build_ydb_params({"$note_id": "n1", "$raw": "bytes"})

    assert params["$note_id"].value == "n1"
    assert params["$note_id"].value_type == ydb.PrimitiveType.Utf8
    assert params["$raw"].value == b"bytes"


def test_json_structures_are_serialised_to_text():
    params = _repository(_FakePool())._build_ydb_params({"$payload": {"a": [1, 2]}})

    assert params["$payload"].value == '{"a": [1, 2]}'
    assert isinstance(params["$payload"].value_type, ydb.OptionalType)


def test_none_is_allowed_only_for_optional_parameters():
    repository = _repository(_FakePool())

    assert repository._build_ydb_params({"$body": None})["$body"].value is None
    with pytest.raises(TypeError, match="not optional"):
        repository._build_ydb_params({"$note_id": None})


def test_undeclared_parameter_is_rejected():
    with pytest.raises(TypeError, match="not declared"):
        _repository(_FakePool())._build_ydb_params({"$unknown": 1})


def test_save_query_declares_every_column_and_upserts_in_sorted_order():
    query = _repository(_FakePool())._build_save_query(_Note("n1", "text", None))

    assert "DECLARE $body AS Utf8?;" in query
    assert "DECLARE $note_id AS Utf8;" in query
    assert "DECLARE $payload AS Json?;" in query
    assert "DECLARE $limit" not in query
    assert "UPSERT INTO notes (body, note_id, payload)" in query
    assert "VALUES ($body, $note_id, $payload);" in query


def test_save_query_rejects_a_mapper_whose_params_do_not_match_its_columns():
    class _BrokenMapper(_NoteMapper):
        def to_ydb_params(self, entity):
            return {"$note_id": entity.note_id}

    repository = YDBRepository(_FakePool(), _BrokenMapper(), "notes")

    with pytest.raises(ValueError, match="Parameter mismatch"):
        repository._build_save_query(_Note("n1", None, None))


async def test_standalone_query_maps_rows_to_entities():
    pool = _FakePool(rows=[{"note_id": "n1", "body": "b", "payload": None}])

    notes = await _repository(pool)._execute_query("SELECT 1;", {"$note_id": "n1"})

    assert notes == [_Note("n1", "b", None)]


async def test_transactional_query_collects_every_streamed_part_without_committing():
    tx = _FakeTx(
        [
            _ResultSet([{"note_id": "n1", "body": None, "payload": None}]),
            _ResultSet([{"note_id": "n2", "body": None, "payload": None}]),
        ]
    )
    pool = _FakePool()

    notes = await _repository(pool)._execute_query("SELECT 1;", {}, tx=tx)

    assert [note.note_id for note in notes] == ["n1", "n2"]
    assert tx.calls[0][2] is False
    assert pool.calls == []


async def test_sdk_errors_surface_as_persistence_error():
    pool = _FakePool(error=ydb.issues.Unavailable("node is down"))

    with pytest.raises(PersistenceError, match="notes"):
        await _repository(pool)._execute_query("SELECT 1;", {})
