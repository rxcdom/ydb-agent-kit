from __future__ import annotations

from typing import Any, Dict, Optional

import ydb

from src.accounts.domain.entities.user import User
from src.accounts.ports.user_repository import UserRepository
from src.shared.domain.value_objects.user_id import UserId
from src.shared.infrastructure.database.ydb.repository import DataMapper, YDBRepository
from src.shared.infrastructure.database.ydb.utils import (
    normalize_datetime_to_utc,
    optional_decode,
    read_row_value,
    safe_decode,
)

TABLE_NAME = "users"


class UserMapper(DataMapper[User]):
    def to_domain(self, row: Any) -> User:
        return User(
            user_id=UserId.from_string(safe_decode(read_row_value(row, "user_id"), "user_id")),
            display_name=optional_decode(read_row_value(row, "display_name"), "display_name"),
            created_at=normalize_datetime_to_utc(read_row_value(row, "created_at")),
        )

    def to_ydb_params(self, entity: User) -> Dict[str, Any]:
        return {
            "$user_id": str(entity.user_id),
            "$display_name": entity.display_name,
            "$created_at": entity.created_at,
        }

    def get_ydb_type_map(self) -> Dict[str, Any]:
        return {
            "$user_id": ydb.PrimitiveType.Utf8,
            "$display_name": ydb.OptionalType(ydb.PrimitiveType.Utf8),
            "$created_at": ydb.PrimitiveType.Timestamp,
        }


class YDBUserRepository(YDBRepository[User], UserRepository):
    def __init__(self, pool: ydb.aio.QuerySessionPool):
        super().__init__(pool, UserMapper(), TABLE_NAME)

    async def save(self, user: User) -> None:
        await self._execute_rows(self._build_save_query(user), self._mapper.to_ydb_params(user))

    async def find_by_id(self, user_id: UserId) -> Optional[User]:
        # Primary-key read: the credential check needs no secondary index.
        query = f"""
        {self._build_declare_block(["$user_id"])}

        SELECT user_id, display_name, created_at
        FROM {TABLE_NAME}
        WHERE user_id = $user_id;
        """
        users = await self._execute_query(query, {"$user_id": str(user_id)})
        return users[0] if users else None
