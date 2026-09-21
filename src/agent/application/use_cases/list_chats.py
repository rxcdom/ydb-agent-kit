from __future__ import annotations

from typing import List

from src.agent.domain.entities.chat import Chat
from src.agent.ports.repository_manager import RepositoryManager
from src.shared.domain.value_objects.user_id import UserId


class ListChatsUseCase:
    """List the chats of the authenticated user, newest first."""

    def __init__(self, repository_manager: RepositoryManager) -> None:
        self._repositories = repository_manager

    async def execute(self, user_id: UserId) -> List[Chat]:
        return await self._repositories.chats.find_by_user_id(user_id)
