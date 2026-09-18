from abc import ABC, abstractmethod
from typing import Optional

from src.accounts.domain.entities.user import User
from src.shared.domain.value_objects.user_id import UserId


class UserRepository(ABC):
    @abstractmethod
    async def save(self, user: User) -> None:
        """Insert or replace the user row."""

    @abstractmethod
    async def find_by_id(self, user_id: UserId) -> Optional[User]:
        """Primary-key read; ``None`` when no such user exists."""
