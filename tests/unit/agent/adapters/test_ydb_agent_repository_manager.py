from datetime import datetime, timezone

import pytest

from src.agent.adapters.persistence.ydb_agent_repository_manager import YDBAgentRepositoryManager
from src.agent.domain.entities.message import Message, MessageRole
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.domain.value_objects.message_id import MessageId
from src.agent.domain.value_objects.message_status import MessageStatus
from src.agent.ports.chat_agent_context_repository import ChatAgentContextRepository
from src.agent.ports.chat_repository import ChatRepository
from src.agent.ports.message_repository import MessageRepository
from src.agent.ports.repository_manager import RepositoryManager
from src.agent.ports.user_memory_repository import UserMemoryRepository
from src.shared.domain.value_objects.user_id import UserId
from tests.unit.agent.adapters.ydb_fakes import FakePool, FakeTx


class _TransactionalPool(FakePool):
    """A pool whose transaction helper runs the callback on one fake context."""

    def __init__(self):
        super().__init__()
        self.tx = FakeTx()
        self.rolled_back = False
        self.tx.rollback = self._rollback

    async def _rollback(self):
        self.rolled_back = True

    async def retry_tx_async(self, callee):
        return await callee(self.tx)


def test_manager_exposes_the_four_repositories_of_the_port():
    manager = YDBAgentRepositoryManager(FakePool())

    assert isinstance(manager, RepositoryManager)
    assert isinstance(manager.chats, ChatRepository)
    assert isinstance(manager.messages, MessageRepository)
    assert isinstance(manager.user_memory, UserMemoryRepository)
    assert isinstance(manager.chat_agent_context, ChatAgentContextRepository)


def test_repositories_are_created_once():
    manager = YDBAgentRepositoryManager(FakePool())

    assert manager.messages is manager.messages
    assert manager.chats is manager.chats


async def test_operations_share_one_transaction_context_and_return_in_order():
    pool = _TransactionalPool()
    manager = YDBAgentRepositoryManager(pool)
    user_message_id = MessageId.generate()
    reply = Message(
        message_id=MessageId.generate(),
        chat_id=ChatId.generate(),
        user_id=UserId.generate(),
        role=MessageRole.ASSISTANT,
        content="Two tasks are overdue.",
        tokens=12,
        status=MessageStatus.SENT,
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )

    results = await manager.execute_in_transaction(
        [
            lambda tx: manager.messages.update_status(user_message_id, MessageStatus.SENT, tx=tx),
            lambda tx: manager.messages.save(reply, tx=tx),
        ]
    )

    assert results == [None, reply]
    assert pool.calls == []
    statements = [query for query, _parameters, _commit in pool.tx.calls]
    assert "UPDATE messages" in statements[0]
    assert "UPSERT INTO messages" in statements[1]
    assert all(commit is False for _query, _parameters, commit in pool.tx.calls)
    assert pool.rolled_back is False


async def test_failing_operation_rolls_the_transaction_back():
    pool = _TransactionalPool()
    manager = YDBAgentRepositoryManager(pool)

    async def failing(_tx):
        raise RuntimeError("the second write failed")

    with pytest.raises(RuntimeError, match="second write failed"):
        await manager.execute_in_transaction([failing])

    assert pool.rolled_back is True
