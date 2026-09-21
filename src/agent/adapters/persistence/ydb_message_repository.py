from __future__ import annotations

from typing import Any, Dict, List, Optional

import ydb

from src.agent.domain.entities.message import Message, MessageRole
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.domain.value_objects.message_id import MessageId
from src.agent.domain.value_objects.message_status import MessageStatus
from src.agent.ports.message_repository import MessageRepository
from src.shared.domain.value_objects.user_id import UserId
from src.shared.infrastructure.database.ydb.repository import DataMapper, YDBRepository
from src.shared.infrastructure.database.ydb.utils import (
    normalize_datetime_to_utc,
    read_row_value,
    safe_decode,
    safe_parse_json,
)

TABLE_NAME = "messages"
CHAT_CREATED_INDEX = "idx_messages_chat_created"

_COLUMNS = "message_id, chat_id, user_id, role, content, tokens, status, trace, created_at"
_COLUMN_PARAMS = (
    "$message_id",
    "$chat_id",
    "$user_id",
    "$role",
    "$content",
    "$tokens",
    "$status",
    "$trace",
    "$created_at",
)


class MessageMapper(DataMapper[Message]):
    def to_domain(self, row: Any) -> Message:
        trace = safe_parse_json(read_row_value(row, "trace"), "trace")
        if trace is not None and not isinstance(trace, dict):
            raise ValueError(
                f"Column 'trace' must hold a JSON object, got {type(trace).__name__}"
            )
        return Message(
            message_id=MessageId.from_string(
                safe_decode(read_row_value(row, "message_id"), "message_id")
            ),
            chat_id=ChatId.from_string(safe_decode(read_row_value(row, "chat_id"), "chat_id")),
            user_id=UserId.from_string(safe_decode(read_row_value(row, "user_id"), "user_id")),
            role=MessageRole(safe_decode(read_row_value(row, "role"), "role")),
            content=safe_decode(read_row_value(row, "content"), "content"),
            tokens=int(read_row_value(row, "tokens")),
            status=MessageStatus(safe_decode(read_row_value(row, "status"), "status")),
            created_at=normalize_datetime_to_utc(read_row_value(row, "created_at")),
            trace=trace,
        )

    def to_ydb_params(self, entity: Message) -> Dict[str, Any]:
        return {
            "$message_id": str(entity.message_id),
            "$chat_id": str(entity.chat_id),
            "$user_id": str(entity.user_id),
            "$role": entity.role.value,
            "$content": entity.content,
            "$tokens": entity.tokens,
            "$status": entity.status.value,
            # Handed over as a dict; the repository base serialises Json parameters.
            "$trace": entity.trace,
            "$created_at": entity.created_at,
        }

    def get_ydb_type_map(self) -> Dict[str, Any]:
        return {
            "$message_id": ydb.PrimitiveType.Utf8,
            "$chat_id": ydb.PrimitiveType.Utf8,
            "$user_id": ydb.PrimitiveType.Utf8,
            "$role": ydb.PrimitiveType.Utf8,
            "$content": ydb.PrimitiveType.Utf8,
            "$tokens": ydb.PrimitiveType.Int32,
            "$status": ydb.PrimitiveType.Utf8,
            "$trace": ydb.OptionalType(ydb.PrimitiveType.Json),
            "$created_at": ydb.PrimitiveType.Timestamp,
            # Query-only parameters.
            "$limit": ydb.PrimitiveType.Uint64,
            "$offset": ydb.PrimitiveType.Uint64,
        }

    def get_column_params(self) -> List[str]:
        return list(_COLUMN_PARAMS)


class YDBMessageRepository(YDBRepository[Message], MessageRepository):
    """``messages`` table.

    Messages of a chat are ordered by ``created_at``; ``message_id`` breaks ties,
    so paging stays stable when two messages share a timestamp.
    """

    def __init__(self, pool: ydb.aio.QuerySessionPool):
        super().__init__(pool, MessageMapper(), TABLE_NAME)

    async def save(self, message: Message, tx: Any = None) -> Message:
        await self._execute_rows(
            self._build_save_query(message), self._mapper.to_ydb_params(message), tx
        )
        return message

    async def find_by_chat_id(
        self,
        chat_id: ChatId,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        tx: Any = None,
    ) -> List[Message]:
        """Return the messages of a chat, oldest first.

        Without ``limit`` no LIMIT clause is rendered and every message comes
        back. YQL has no OFFSET without LIMIT, so ``offset`` alone is rejected.
        """
        if offset is not None and limit is None:
            raise ValueError("offset requires a limit")

        params: Dict[str, Any] = {"$chat_id": str(chat_id)}
        # Uses idx_messages_chat_created: the chat filter and the ordering follow its key.
        clauses = [
            f"SELECT {_COLUMNS}",
            f"FROM {TABLE_NAME} VIEW {CHAT_CREATED_INDEX}",
            "WHERE chat_id = $chat_id",
            "ORDER BY created_at ASC, message_id ASC",
        ]
        if limit is not None:
            params["$limit"] = limit
            clauses.append("LIMIT $limit")
        if offset is not None:
            params["$offset"] = offset
            clauses.append("OFFSET $offset")

        query = self._build_declare_block(list(params)) + "\n\n" + "\n".join(clauses) + ";"
        return await self._execute_query(query, params, tx)

    async def find_recent_by_chat_id(self, chat_id: ChatId, limit: int) -> List[Message]:
        # Uses idx_messages_chat_created. The newest rows are selected in descending
        # order and turned around, so the caller reads them oldest first.
        query = f"""
        {self._build_declare_block(["$chat_id", "$limit"])}

        SELECT {_COLUMNS}
        FROM {TABLE_NAME} VIEW {CHAT_CREATED_INDEX}
        WHERE chat_id = $chat_id
        ORDER BY created_at DESC, message_id DESC
        LIMIT $limit;
        """
        newest_first = await self._execute_query(
            query, {"$chat_id": str(chat_id), "$limit": limit}
        )
        return list(reversed(newest_first))

    async def count_by_chat_id(self, chat_id: ChatId) -> int:
        # Uses idx_messages_chat_created: counting reads the index rows of the chat only.
        query = f"""
        {self._build_declare_block(["$chat_id"])}

        SELECT COUNT(*) AS total
        FROM {TABLE_NAME} VIEW {CHAT_CREATED_INDEX}
        WHERE chat_id = $chat_id;
        """
        rows = await self._execute_rows(query, {"$chat_id": str(chat_id)})
        return int(read_row_value(rows[0], "total")) if rows else 0

    async def update_status(
        self, message_id: MessageId, status: MessageStatus, tx: Any = None
    ) -> None:
        # Primary-key update: no secondary index is involved.
        query = f"""
        {self._build_declare_block(["$message_id", "$status"])}

        UPDATE {TABLE_NAME}
        SET status = $status
        WHERE message_id = $message_id;
        """
        await self._execute_rows(
            query, {"$message_id": str(message_id), "$status": status.value}, tx
        )
