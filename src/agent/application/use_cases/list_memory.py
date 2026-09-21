from __future__ import annotations

from typing import List

from src.agent.application.memory.user_memory_writer import USER_MEMORY_MAX_ENTRIES
from src.agent.domain.entities.user_memory import UserMemory
from src.agent.ports.repository_manager import RepositoryManager
from src.shared.domain.value_objects.user_id import UserId


class ListMemoryUseCase:
    """List what the agent has remembered about the authenticated user.

    It makes the long-term memory observable outside the chat: the vault can be
    inspected without asking the agent to recall it.
    """

    def __init__(self, repository_manager: RepositoryManager) -> None:
        self._repositories = repository_manager

    async def execute(self, user_id: UserId) -> List[UserMemory]:
        """Return the user's entries, most recently updated first.

        The vault never holds more than its cap, so one read returns all of it.
        """
        return await self._repositories.user_memory.find_by_user_id(
            user_id, limit=USER_MEMORY_MAX_ENTRIES
        )
