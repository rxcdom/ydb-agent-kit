from __future__ import annotations

from typing import Any, List, Sequence

import ydb

from src.agent.adapters.persistence.ydb_chat_agent_context_repository import (
    YDBChatAgentContextRepository,
)
from src.agent.adapters.persistence.ydb_chat_repository import YDBChatRepository
from src.agent.adapters.persistence.ydb_message_repository import YDBMessageRepository
from src.agent.adapters.persistence.ydb_user_memory_repository import YDBUserMemoryRepository
from src.agent.ports.repository_manager import RepositoryManager, TransactionalOperation
from src.shared.infrastructure.database.ydb.transaction_manager import YDBTransactionManager


class YDBAgentRepositoryManager(RepositoryManager):
    """The agent module's repositories on one session pool, plus their transactions.

    An operation passed to ``execute_in_transaction`` receives the transaction
    context and hands it on as ``tx`` to the repository methods it calls; those
    statements then run inside the transaction instead of committing on their own.
    """

    def __init__(self, pool: ydb.aio.QuerySessionPool):
        self._transaction_manager = YDBTransactionManager(pool)
        self._chats = YDBChatRepository(pool)
        self._messages = YDBMessageRepository(pool)
        self._user_memory = YDBUserMemoryRepository(pool)
        self._chat_agent_context = YDBChatAgentContextRepository(pool)

    @property
    def chats(self) -> YDBChatRepository:
        return self._chats

    @property
    def messages(self) -> YDBMessageRepository:
        return self._messages

    @property
    def user_memory(self) -> YDBUserMemoryRepository:
        return self._user_memory

    @property
    def chat_agent_context(self) -> YDBChatAgentContextRepository:
        return self._chat_agent_context

    async def execute_in_transaction(
        self, operations: Sequence[TransactionalOperation]
    ) -> List[Any]:
        return await self._transaction_manager.execute_in_transaction(operations)
