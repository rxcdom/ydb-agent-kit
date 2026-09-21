from __future__ import annotations

from typing import Any, Dict, List, Optional

import ydb

from src.agent.domain.entities.chat import Chat
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.ports.chat_repository import ChatRepository
from src.shared.domain.value_objects.user_id import UserId
from src.shared.infrastructure.database.ydb.repository import DataMapper, YDBRepository
from src.shared.infrastructure.database.ydb.utils import (
    normalize_datetime_to_utc,
    optional_decode,
    read_row_value,
    safe_decode,
)

TABLE_NAME = "chats"
USER_INDEX = "idx_chats_user_id"

_COLUMNS = "chat_id, user_id, title, created_at, updated_at"


class ChatMapper(DataMapper[Chat]):
    def to_domain(self, row: Any) -> Chat:
        return Chat(
            chat_id=ChatId.from_string(safe_decode(read_row_value(row, "chat_id"), "chat_id")),
            user_id=UserId.from_string(safe_decode(read_row_value(row, "user_id"), "user_id")),
            title=optional_decode(read_row_value(row, "title"), "title"),
            created_at=normalize_datetime_to_utc(read_row_value(row, "created_at")),
            updated_at=normalize_datetime_to_utc(read_row_value(row, "updated_at")),
        )

    def to_ydb_params(self, entity: Chat) -> Dict[str, Any]:
        return {
            "$chat_id": str(entity.chat_id),
            "$user_id": str(entity.user_id),
            "$title": entity.title,
            "$created_at": entity.created_at,
            "$updated_at": entity.updated_at,
        }

    def get_ydb_type_map(self) -> Dict[str, Any]:
        return {
            "$chat_id": ydb.PrimitiveType.Utf8,
            "$user_id": ydb.PrimitiveType.Utf8,
            "$title": ydb.OptionalType(ydb.PrimitiveType.Utf8),
            "$created_at": ydb.PrimitiveType.Timestamp,
            "$updated_at": ydb.PrimitiveType.Timestamp,
        }


class YDBChatRepository(YDBRepository[Chat], ChatRepository):
    """``chats`` table. The timestamps are set by the caller, never here."""

    def __init__(self, pool: ydb.aio.QuerySessionPool):
        super().__init__(pool, ChatMapper(), TABLE_NAME)

    async def save(self, chat: Chat, tx: Any = None) -> Chat:
        await self._execute_rows(
            self._build_save_query(chat), self._mapper.to_ydb_params(chat), tx
        )
        return chat

    async def find_by_id(self, chat_id: ChatId) -> Optional[Chat]:
        # Primary-key read: no secondary index is involved.
        query = f"""
        {self._build_declare_block(["$chat_id"])}

        SELECT {_COLUMNS}
        FROM {TABLE_NAME}
        WHERE chat_id = $chat_id;
        """
        chats = await self._execute_query(query, {"$chat_id": str(chat_id)})
        return chats[0] if chats else None

    async def find_by_user_id(self, user_id: UserId) -> List[Chat]:
        # Uses idx_chats_user_id: without VIEW the owner filter would scan the table.
        query = f"""
        {self._build_declare_block(["$user_id"])}

        SELECT {_COLUMNS}
        FROM {TABLE_NAME} VIEW {USER_INDEX}
        WHERE user_id = $user_id
        ORDER BY created_at DESC;
        """
        return await self._execute_query(query, {"$user_id": str(user_id)})
