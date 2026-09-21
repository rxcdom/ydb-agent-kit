from __future__ import annotations

from typing import Any, Dict, Optional

import ydb

from src.agent.domain.entities.chat_agent_context import ChatAgentContext
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.ports.chat_agent_context_repository import ChatAgentContextRepository
from src.shared.domain.value_objects.user_id import UserId
from src.shared.infrastructure.database.ydb.repository import DataMapper, YDBRepository
from src.shared.infrastructure.database.ydb.utils import (
    normalize_datetime_to_utc,
    optional_decode,
    read_row_value,
    safe_decode,
)

TABLE_NAME = "chat_agent_context"

_COLUMNS = (
    "chat_id, user_id, last_tool, last_window_from, last_window_to, "
    "last_date_field, last_project, created_at, updated_at"
)
_OPTIONAL_TEXT_COLUMNS = (
    "last_tool",
    "last_window_from",
    "last_window_to",
    "last_date_field",
    "last_project",
)


class ChatAgentContextMapper(DataMapper[ChatAgentContext]):
    def to_domain(self, row: Any) -> ChatAgentContext:
        helper_values = {
            column: optional_decode(read_row_value(row, column), column)
            for column in _OPTIONAL_TEXT_COLUMNS
        }
        return ChatAgentContext(
            chat_id=ChatId.from_string(safe_decode(read_row_value(row, "chat_id"), "chat_id")),
            user_id=UserId.from_string(safe_decode(read_row_value(row, "user_id"), "user_id")),
            created_at=normalize_datetime_to_utc(read_row_value(row, "created_at")),
            updated_at=normalize_datetime_to_utc(read_row_value(row, "updated_at")),
            **helper_values,
        )

    def to_ydb_params(self, entity: ChatAgentContext) -> Dict[str, Any]:
        return {
            "$chat_id": str(entity.chat_id),
            "$user_id": str(entity.user_id),
            "$last_tool": entity.last_tool,
            "$last_window_from": entity.last_window_from,
            "$last_window_to": entity.last_window_to,
            "$last_date_field": entity.last_date_field,
            "$last_project": entity.last_project,
            "$created_at": entity.created_at,
            "$updated_at": entity.updated_at,
        }

    def get_ydb_type_map(self) -> Dict[str, Any]:
        optional_text = ydb.OptionalType(ydb.PrimitiveType.Utf8)
        return {
            "$chat_id": ydb.PrimitiveType.Utf8,
            "$user_id": ydb.PrimitiveType.Utf8,
            "$last_tool": optional_text,
            "$last_window_from": optional_text,
            "$last_window_to": optional_text,
            "$last_date_field": optional_text,
            "$last_project": optional_text,
            "$created_at": ydb.PrimitiveType.Timestamp,
            "$updated_at": ydb.PrimitiveType.Timestamp,
        }


class YDBChatAgentContextRepository(YDBRepository[ChatAgentContext], ChatAgentContextRepository):
    """``chat_agent_context`` table: one row per chat, keyed by ``chat_id``."""

    def __init__(self, pool: ydb.aio.QuerySessionPool):
        super().__init__(pool, ChatAgentContextMapper(), TABLE_NAME)

    async def save(self, context: ChatAgentContext, tx: Any = None) -> ChatAgentContext:
        await self._execute_rows(
            self._build_save_query(context), self._mapper.to_ydb_params(context), tx
        )
        return context

    async def find_by_chat_id(self, chat_id: ChatId) -> Optional[ChatAgentContext]:
        # Primary-key read: the table has no secondary index and needs none.
        query = f"""
        {self._build_declare_block(["$chat_id"])}

        SELECT {_COLUMNS}
        FROM {TABLE_NAME}
        WHERE chat_id = $chat_id;
        """
        contexts = await self._execute_query(query, {"$chat_id": str(chat_id)})
        return contexts[0] if contexts else None
