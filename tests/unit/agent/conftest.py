"""Fixtures shared by the agent module's unit tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.ports.chat_agent_context_repository import ChatAgentContextRepository
from src.agent.ports.chat_repository import ChatRepository
from src.agent.ports.message_repository import MessageRepository
from src.agent.ports.repository_manager import RepositoryManager
from src.agent.ports.user_memory_repository import UserMemoryRepository


@pytest.fixture
def transaction() -> object:
    """The placeholder the manager double hands to transactional operations."""
    return object()


@pytest.fixture
def mock_repository_manager(transaction: object) -> MagicMock:
    """A ``RepositoryManager`` double with its four repositories.

    Every repository is specced on its port, so a call to a method the port does
    not declare fails the test. The defaults describe an empty datastore; a test
    overrides the return values it cares about. ``execute_in_transaction`` runs
    the operations in order with the ``transaction`` placeholder and returns
    their results, like the real manager does.
    """
    manager = MagicMock(spec=RepositoryManager)

    manager.chats = AsyncMock(spec=ChatRepository)
    manager.chats.find_by_id.return_value = None
    manager.chats.find_by_user_id.return_value = []

    manager.messages = AsyncMock(spec=MessageRepository)
    manager.messages.find_by_chat_id.return_value = []
    manager.messages.find_recent_by_chat_id.return_value = []
    manager.messages.count_by_chat_id.return_value = 0

    manager.user_memory = AsyncMock(spec=UserMemoryRepository)
    manager.user_memory.find_by_user_id.return_value = []
    manager.user_memory.search_by_user_id.return_value = []
    manager.user_memory.count_by_user_id.return_value = 0
    manager.user_memory.find_oldest_by_user_id.return_value = None

    manager.chat_agent_context = AsyncMock(spec=ChatAgentContextRepository)
    manager.chat_agent_context.find_by_chat_id.return_value = None

    async def run_operations(operations):
        return [await operation(transaction) for operation in operations]

    manager.execute_in_transaction = AsyncMock(side_effect=run_operations)
    return manager
