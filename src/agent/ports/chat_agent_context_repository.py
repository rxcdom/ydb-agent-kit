from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.agent.domain.entities.chat_agent_context import ChatAgentContext
from src.agent.domain.value_objects.chat_id import ChatId


class ChatAgentContextRepository(ABC):
    """Storage of the per-chat conversation state (one row per chat)."""

    @abstractmethod
    async def save(self, context: ChatAgentContext, tx: Any = None) -> ChatAgentContext:
        """Insert the row, or replace the stored row of the same chat.

        ``tx`` is an optional transaction context handed out by
        ``RepositoryManager.execute_in_transaction``.
        """

    @abstractmethod
    async def find_by_chat_id(self, chat_id: ChatId) -> Optional[ChatAgentContext]:
        """Return the state of the chat, or ``None`` when none was recorded yet."""
