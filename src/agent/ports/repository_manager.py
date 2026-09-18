from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, List, Sequence

from src.agent.ports.chat_agent_context_repository import ChatAgentContextRepository
from src.agent.ports.chat_repository import ChatRepository
from src.agent.ports.message_repository import MessageRepository
from src.agent.ports.user_memory_repository import UserMemoryRepository

# An operation receives the transaction context and passes it on as ``tx`` to the
# repository methods it calls.
TransactionalOperation = Callable[[Any], Awaitable[Any]]


class RepositoryManager(ABC):
    """Single access point to the agent module's repositories and transactions."""

    @property
    @abstractmethod
    def chats(self) -> ChatRepository:
        """Repository of chats."""

    @property
    @abstractmethod
    def messages(self) -> MessageRepository:
        """Repository of chat messages."""

    @property
    @abstractmethod
    def user_memory(self) -> UserMemoryRepository:
        """Repository of the long-term memory vault."""

    @property
    @abstractmethod
    def chat_agent_context(self) -> ChatAgentContextRepository:
        """Repository of the per-chat conversation state."""

    @abstractmethod
    async def execute_in_transaction(
        self, operations: Sequence[TransactionalOperation]
    ) -> List[Any]:
        """Run ``operations`` in order as one atomic transaction.

        Nothing is committed when any operation raises. The results are returned
        in the order of the operations. The transaction may be replayed on a
        transient failure, so operations have to be safe to run more than once.
        """
