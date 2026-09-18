"""Port of the long-term memory vault (``user_memory`` table).

Every method is scoped by ``user_id``. The vault is strictly per user, and the
owner id always comes from the authenticated request, never from model output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.agent.domain.entities.user_memory import UserMemory
from src.shared.domain.value_objects.user_id import UserId


class UserMemoryRepository(ABC):
    """Storage of long-term memory entries."""

    @abstractmethod
    async def save(self, memory: UserMemory) -> UserMemory:
        """Insert the entry, or replace the stored row with the same ``memory_id``."""

    @abstractmethod
    async def find_by_user_id(self, user_id: UserId, *, limit: int) -> List[UserMemory]:
        """Return the user's entries, most recently updated first, up to ``limit``."""

    @abstractmethod
    async def search_by_user_id(
        self, user_id: UserId, *, query: str, limit: int
    ) -> List[UserMemory]:
        """Return entries whose ``content`` or ``topic`` contains ``query``.

        The match is a case-insensitive substring match and ``query`` is taken
        literally (pattern wildcards are escaped). Results are ordered most
        recently updated first, up to ``limit``.
        """

    @abstractmethod
    async def delete(self, memory_id: str, user_id: UserId) -> None:
        """Delete one entry. The ``user_id`` guard rules out cross-user deletes."""

    @abstractmethod
    async def delete_all_by_user_id(self, user_id: UserId) -> None:
        """Delete every entry the user owns (used when a workspace is reset)."""

    @abstractmethod
    async def count_by_user_id(self, user_id: UserId) -> int:
        """Return how many entries the user has (used to enforce the vault cap)."""

    @abstractmethod
    async def find_oldest_by_user_id(self, user_id: UserId) -> Optional[UserMemory]:
        """Return the entry with the earliest ``created_at``: the eviction candidate."""
