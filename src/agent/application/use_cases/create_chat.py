from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from src.agent.domain.entities.chat import Chat
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.ports.repository_manager import RepositoryManager
from src.shared.domain.exceptions import ValidationError
from src.shared.domain.value_objects.user_id import UserId


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CreateChatUseCase:
    """Open a new chat owned by the authenticated user."""

    def __init__(
        self,
        repository_manager: RepositoryManager,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._repositories = repository_manager
        self._clock = clock

    async def execute(self, user_id: UserId, title: Optional[str] = None) -> Chat:
        """Create and store the chat. An unacceptable title is a ``ValidationError``."""
        now = self._clock()
        try:
            chat = Chat(
                chat_id=ChatId.generate(),
                user_id=user_id,
                created_at=now,
                updated_at=now,
                title=title,
            )
        except ValueError as error:
            raise ValidationError(str(error)) from error

        await self._repositories.chats.save(chat)
        return chat
