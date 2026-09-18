from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from src.accounts.domain.entities.user import User
from src.accounts.ports.user_repository import UserRepository
from src.shared.domain.value_objects.user_id import UserId


class CreateUserUseCase:
    """Creates a user whose generated id is also its bearer credential."""

    def __init__(
        self,
        user_repository: UserRepository,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._user_repository = user_repository
        self._clock = clock

    async def execute(self, display_name: Optional[str] = None) -> User:
        user = User(user_id=UserId.generate(), display_name=display_name, created_at=self._clock())
        await self._user_repository.save(user)
        return user
