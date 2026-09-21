from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.agent.domain.entities.message import Message
from src.agent.domain.exceptions import ChatAccessDeniedError, ChatNotFoundError
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.ports.repository_manager import RepositoryManager
from src.shared.domain.exceptions import ValidationError
from src.shared.domain.value_objects.user_id import UserId

DEFAULT_HISTORY_PAGE_SIZE = 50
MAX_HISTORY_PAGE_SIZE = 100


@dataclass(frozen=True)
class ChatHistoryPage:
    """One page of a chat, oldest message first.

    ``total_count`` is the number of messages in the whole chat; ``has_more``
    tells whether messages exist after this page.
    """

    messages: List[Message]
    total_count: int
    has_more: bool


class GetChatHistoryUseCase:
    """Page through the messages of a chat the authenticated user owns."""

    def __init__(self, repository_manager: RepositoryManager) -> None:
        self._repositories = repository_manager

    async def execute(
        self,
        user_id: UserId,
        chat_id: ChatId,
        limit: int = DEFAULT_HISTORY_PAGE_SIZE,
        offset: int = 0,
    ) -> ChatHistoryPage:
        """Return ``limit`` messages starting ``offset`` messages into the chat.

        Raises ``ChatNotFoundError`` for a missing chat, ``ChatAccessDeniedError``
        for a chat of another owner, and ``ValidationError`` for paging values
        out of range.
        """
        if not 1 <= limit <= MAX_HISTORY_PAGE_SIZE:
            raise ValidationError(f"limit must be between 1 and {MAX_HISTORY_PAGE_SIZE}")
        if offset < 0:
            raise ValidationError("offset must not be negative")

        chat = await self._repositories.chats.find_by_id(chat_id)
        if chat is None:
            raise ChatNotFoundError(f"Chat {chat_id} not found")
        if chat.user_id != user_id:
            raise ChatAccessDeniedError(f"Chat {chat_id} belongs to another user")

        total_count = await self._repositories.messages.count_by_chat_id(chat_id)
        messages = await self._repositories.messages.find_by_chat_id(
            chat_id, limit=limit, offset=offset
        )
        return ChatHistoryPage(
            messages=messages,
            total_count=total_count,
            has_more=offset + len(messages) < total_count,
        )
