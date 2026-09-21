from __future__ import annotations

from typing import Any, Dict, List, Optional

import ydb

from src.agent.domain.entities.user_memory import UserMemory
from src.agent.ports.user_memory_repository import UserMemoryRepository
from src.shared.domain.value_objects.user_id import UserId
from src.shared.infrastructure.database.ydb.repository import DataMapper, YDBRepository
from src.shared.infrastructure.database.ydb.utils import (
    normalize_datetime_to_utc,
    optional_decode,
    read_row_value,
    safe_decode,
)

TABLE_NAME = "user_memory"
USER_INDEX = "idx_user_memory_user_id"

# YQL accepts neither "%", "_" nor a backslash as the ESCAPE character.
LIKE_ESCAPE_CHARACTER = "!"

_COLUMNS = "memory_id, user_id, content, topic, created_at, updated_at"
_COLUMN_PARAMS = (
    "$memory_id",
    "$user_id",
    "$content",
    "$topic",
    "$created_at",
    "$updated_at",
)


def escape_like_substring(value: str) -> str:
    """Escape LIKE wildcards so ``value`` is matched as a literal substring."""
    escape = LIKE_ESCAPE_CHARACTER
    return (
        value.replace(escape, escape + escape)
        .replace("%", escape + "%")
        .replace("_", escape + "_")
    )


class UserMemoryMapper(DataMapper[UserMemory]):
    def to_domain(self, row: Any) -> UserMemory:
        return UserMemory(
            memory_id=safe_decode(read_row_value(row, "memory_id"), "memory_id"),
            user_id=UserId.from_string(safe_decode(read_row_value(row, "user_id"), "user_id")),
            content=safe_decode(read_row_value(row, "content"), "content"),
            topic=optional_decode(read_row_value(row, "topic"), "topic"),
            created_at=normalize_datetime_to_utc(read_row_value(row, "created_at")),
            updated_at=normalize_datetime_to_utc(read_row_value(row, "updated_at")),
        )

    def to_ydb_params(self, entity: UserMemory) -> Dict[str, Any]:
        return {
            "$memory_id": entity.memory_id,
            "$user_id": str(entity.user_id),
            "$content": entity.content,
            "$topic": entity.topic,
            "$created_at": entity.created_at,
            "$updated_at": entity.updated_at,
        }

    def get_ydb_type_map(self) -> Dict[str, Any]:
        return {
            "$memory_id": ydb.PrimitiveType.Utf8,
            "$user_id": ydb.PrimitiveType.Utf8,
            "$content": ydb.PrimitiveType.Utf8,
            "$topic": ydb.OptionalType(ydb.PrimitiveType.Utf8),
            "$created_at": ydb.PrimitiveType.Timestamp,
            "$updated_at": ydb.PrimitiveType.Timestamp,
            # Query-only parameters.
            "$pattern": ydb.PrimitiveType.Utf8,
            "$limit": ydb.PrimitiveType.Uint64,
        }

    def get_column_params(self) -> List[str]:
        return list(_COLUMN_PARAMS)


class YDBUserMemoryRepository(YDBRepository[UserMemory], UserMemoryRepository):
    """``user_memory`` table. Every statement is scoped by the owner's ``user_id``."""

    def __init__(self, pool: ydb.aio.QuerySessionPool):
        super().__init__(pool, UserMemoryMapper(), TABLE_NAME)

    async def save(self, memory: UserMemory) -> UserMemory:
        await self._execute_rows(
            self._build_save_query(memory), self._mapper.to_ydb_params(memory)
        )
        return memory

    async def find_by_user_id(self, user_id: UserId, *, limit: int) -> List[UserMemory]:
        # Uses idx_user_memory_user_id.
        query = f"""
        {self._build_declare_block(["$user_id", "$limit"])}

        SELECT {_COLUMNS}
        FROM {TABLE_NAME} VIEW {USER_INDEX}
        WHERE user_id = $user_id
        ORDER BY updated_at DESC
        LIMIT $limit;
        """
        return await self._execute_query(query, {"$user_id": str(user_id), "$limit": limit})

    async def search_by_user_id(
        self, user_id: UserId, *, query: str, limit: int
    ) -> List[UserMemory]:
        # Uses idx_user_memory_user_id to reach the owner's rows; the substring
        # match runs over those rows only. Both sides are lower-cased by the same
        # function, which works on Utf8 columns directly. A row without a topic
        # is matched on its content alone.
        yql = f"""
        {self._build_declare_block(["$user_id", "$pattern", "$limit"])}

        SELECT {_COLUMNS}
        FROM {TABLE_NAME} VIEW {USER_INDEX}
        WHERE user_id = $user_id
          AND (
            Unicode::ToLower(content) LIKE Unicode::ToLower($pattern)
              ESCAPE '{LIKE_ESCAPE_CHARACTER}'
            OR (
              topic IS NOT NULL
              AND Unicode::ToLower(topic) LIKE Unicode::ToLower($pattern)
                ESCAPE '{LIKE_ESCAPE_CHARACTER}'
            )
          )
        ORDER BY updated_at DESC
        LIMIT $limit;
        """
        params = {
            "$user_id": str(user_id),
            "$pattern": f"%{escape_like_substring(query)}%",
            "$limit": limit,
        }
        return await self._execute_query(yql, params)

    async def delete(self, memory_id: str, user_id: UserId) -> None:
        # Primary-key delete. The owner condition makes it a no-op for a row that
        # belongs to somebody else.
        query = f"""
        {self._build_declare_block(["$memory_id", "$user_id"])}

        DELETE FROM {TABLE_NAME}
        WHERE memory_id = $memory_id AND user_id = $user_id;
        """
        await self._execute_rows(query, {"$memory_id": memory_id, "$user_id": str(user_id)})

    async def delete_all_by_user_id(self, user_id: UserId) -> None:
        # Uses idx_user_memory_user_id: the keys to delete are selected through
        # the index, because a plain DELETE ... WHERE cannot name one.
        query = f"""
        {self._build_declare_block(["$user_id"])}

        DELETE FROM {TABLE_NAME} ON
        SELECT memory_id
        FROM {TABLE_NAME} VIEW {USER_INDEX}
        WHERE user_id = $user_id;
        """
        await self._execute_rows(query, {"$user_id": str(user_id)})

    async def count_by_user_id(self, user_id: UserId) -> int:
        # Uses idx_user_memory_user_id.
        query = f"""
        {self._build_declare_block(["$user_id"])}

        SELECT COUNT(*) AS total
        FROM {TABLE_NAME} VIEW {USER_INDEX}
        WHERE user_id = $user_id;
        """
        rows = await self._execute_rows(query, {"$user_id": str(user_id)})
        return int(read_row_value(rows[0], "total")) if rows else 0

    async def find_oldest_by_user_id(self, user_id: UserId) -> Optional[UserMemory]:
        # Uses idx_user_memory_user_id.
        query = f"""
        {self._build_declare_block(["$user_id"])}

        SELECT {_COLUMNS}
        FROM {TABLE_NAME} VIEW {USER_INDEX}
        WHERE user_id = $user_id
        ORDER BY created_at ASC
        LIMIT 1;
        """
        memories = await self._execute_query(query, {"$user_id": str(user_id)})
        return memories[0] if memories else None
