"""Generic repository infrastructure for the YDB persistence layer.

``DataMapper`` converts between result rows and domain entities and declares
the YDB type of every query parameter. ``YDBRepository`` executes queries,
builds typed parameters from that declaration, and maps the rows back.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

import ydb

from src.shared.domain.exceptions import PersistenceError
from src.shared.infrastructure.database.ydb.types import format_ydb_type
from src.shared.infrastructure.database.ydb.utils import collect_result_set_rows

T = TypeVar("T")

_JSON_TYPES = (ydb.PrimitiveType.Json, ydb.PrimitiveType.JsonDocument)


class DataMapper(ABC, Generic[T]):
    """Converts between YDB rows and one domain entity type."""

    @abstractmethod
    def to_domain(self, row: Any) -> T:
        """Build the entity from a result row."""

    @abstractmethod
    def to_ydb_params(self, entity: T) -> Dict[str, Any]:
        """Return ``{"$column": python_value}`` for every column of the entity."""

    @abstractmethod
    def get_ydb_type_map(self) -> Dict[str, Any]:
        """Return ``{"$parameter": ydb_type}`` for every parameter the repository uses.

        Example: ``{"$user_id": ydb.PrimitiveType.Utf8,
        "$due_at": ydb.OptionalType(ydb.PrimitiveType.Timestamp)}``.
        Query-only parameters (limits, search patterns) are declared here too.
        """

    def get_column_params(self) -> List[str]:
        """Parameters that correspond to table columns; defaults to all of them."""
        return list(self.get_ydb_type_map().keys())


class YDBRepository(Generic[T]):
    """Base class of the concrete repositories."""

    def __init__(self, pool: ydb.aio.QuerySessionPool, mapper: DataMapper[T], table_path: str):
        self._pool = pool
        self._mapper = mapper
        self._table_path = table_path

    async def _execute_rows(
        self,
        query: str,
        params: Dict[str, Any],
        tx: Optional[ydb.aio.QueryTxContext] = None,
    ) -> List[Any]:
        """Run a query and return the raw rows of its first result set.

        Inside a transaction the statement runs on the given context without
        committing. Standalone statements go through the pool's retry helper.
        Every SDK failure surfaces as ``PersistenceError``; this is the single
        place where that mapping happens for repositories.
        """
        ydb_params = self._build_ydb_params(params)
        try:
            if tx is not None:
                stream = await tx.execute(query, parameters=ydb_params, commit_tx=False)
                result_sets = [result_set async for result_set in stream]
            else:
                result_sets = await self._pool.execute_with_retries(query, parameters=ydb_params)
        except ydb.Error as error:
            raise PersistenceError(f"YDB query on '{self._table_path}' failed: {error}") from error
        return collect_result_set_rows(result_sets)

    async def _execute_query(
        self,
        query: str,
        params: Dict[str, Any],
        tx: Optional[ydb.aio.QueryTxContext] = None,
    ) -> List[T]:
        """Run a query and map every row to a domain entity."""
        rows = await self._execute_rows(query, params, tx)
        return [self._mapper.to_domain(row) for row in rows]

    def _build_ydb_params(self, params: Dict[str, Any]) -> Dict[str, ydb.TypedValue]:
        """Attach the declared YDB type to every parameter value."""
        type_map = self._mapper.get_ydb_type_map()
        ydb_params: Dict[str, ydb.TypedValue] = {}

        for key, value in params.items():
            if key not in type_map:
                raise TypeError(f"Parameter '{key}' is not declared in the mapper's type map.")

            declared_type = type_map[key]
            is_optional = isinstance(declared_type, ydb.OptionalType)
            base_type = declared_type.item if is_optional else declared_type

            if value is None:
                if not is_optional:
                    raise TypeError(f"Parameter '{key}' is not optional but received None.")
            elif base_type == ydb.PrimitiveType.String:
                value = value.encode("utf-8") if isinstance(value, str) else value
            elif base_type in _JSON_TYPES:
                # The SDK expects JSON as text; structures are serialised here
                # so mappers can hand over plain dicts and lists.
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                elif not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)

            ydb_params[key] = ydb.TypedValue(value, declared_type)

        return ydb_params

    def _build_declare_block(self, param_names: List[str]) -> str:
        """Render ``DECLARE`` statements for the given parameters, sorted by name."""
        type_map = self._mapper.get_ydb_type_map()
        lines = []
        for name in sorted(param_names):
            if name not in type_map:
                raise TypeError(f"Parameter '{name}' is not declared in the mapper's type map.")
            lines.append(f"DECLARE {name} AS {format_ydb_type(type_map[name])};")
        return "\n".join(lines)

    def _build_save_query(self, entity: T) -> str:
        """Render the ``UPSERT`` of one entity from the mapper's declarations."""
        params = self._mapper.to_ydb_params(entity)
        expected = set(self._mapper.get_column_params())

        if set(params.keys()) != expected:
            missing = sorted(expected - set(params.keys()))
            extra = sorted(set(params.keys()) - expected)
            raise ValueError(f"Parameter mismatch: missing={missing}, extra={extra}")

        ordered = sorted(params.keys())
        columns = ", ".join(name.removeprefix("$") for name in ordered)
        placeholders = ", ".join(ordered)

        return (
            f"{self._build_declare_block(ordered)}\n\n"
            f"UPSERT INTO {self._table_path} ({columns})\n"
            f"VALUES ({placeholders});"
        )
