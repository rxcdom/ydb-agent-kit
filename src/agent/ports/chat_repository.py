from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from src.agent.domain.entities.chat import Chat
from src.agent.domain.value_objects.chat_id import ChatId
from src.shared.domain.value_objects.user_id import UserId


class ChatRepository(ABC):
    """Storage of chats.

    ``tx`` is an optional transaction context handed out by
    ``RepositoryManager.execute_in_transaction``; its concrete type belongs to the
    persistence adapter.
    """

    @abstractmethod
    async def save(self, chat: Chat, tx: Any = None) -> Chat:
        """Insert the chat, or replace the stored row with the same ``chat_id``."""

    @abstractmethod
    async def find_by_id(self, chat_id: ChatId) -> Optional[Chat]:
        """Return the chat, or ``None`` when it does not exist.

        The lookup is not owner-scoped: the caller compares ``chat.user_id`` with
        the principal, which lets it tell a missing chat from a foreign one.
        """

    @abstractmethod
    async def find_by_user_id(self, user_id: UserId) -> List[Chat]:
        """Return every chat the user owns, newest first by ``created_at``."""
