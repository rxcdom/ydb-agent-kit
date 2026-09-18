from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from src.agent.domain.entities.message import Message
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.domain.value_objects.message_id import MessageId
from src.agent.domain.value_objects.message_status import MessageStatus


class MessageRepository(ABC):
    """Storage of chat messages.

    ``tx`` is an optional transaction context handed out by
    ``RepositoryManager.execute_in_transaction``; its concrete type belongs to the
    persistence adapter. A call that receives it runs inside that transaction.
    """

    @abstractmethod
    async def save(self, message: Message, tx: Any = None) -> Message:
        """Insert the message, or replace the stored row with the same id."""

    @abstractmethod
    async def find_by_chat_id(
        self,
        chat_id: ChatId,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        tx: Any = None,
    ) -> List[Message]:
        """Return the messages of a chat, oldest first.

        ``limit`` and ``offset`` page through the chat from its first message.
        Without them every message is returned.
        """

    @abstractmethod
    async def find_recent_by_chat_id(self, chat_id: ChatId, limit: int) -> List[Message]:
        """Return the most recent ``limit`` messages of a chat, oldest first.

        Unlike ``find_by_chat_id`` with a limit, which starts at the beginning of
        the chat, this selects the tail of the conversation: the last element is
        the newest message. The agent loop builds its history from it.
        """

    @abstractmethod
    async def count_by_chat_id(self, chat_id: ChatId) -> int:
        """Return the number of messages in the chat."""

    @abstractmethod
    async def update_status(
        self, message_id: MessageId, status: MessageStatus, tx: Any = None
    ) -> None:
        """Set the status of one stored message, leaving its other fields as is."""
