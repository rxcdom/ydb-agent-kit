"""In-memory stand-ins for the agent module's outbound ports, used by the gateway tests."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.agent.domain.entities.chat import Chat
from src.agent.domain.entities.chat_agent_context import ChatAgentContext
from src.agent.domain.entities.message import Message
from src.agent.domain.entities.user_memory import UserMemory
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.domain.value_objects.message_id import MessageId
from src.agent.domain.value_objects.message_status import MessageStatus
from src.agent.ports.chat_agent_context_repository import ChatAgentContextRepository
from src.agent.ports.chat_repository import ChatRepository
from src.agent.ports.llm_client import LLMClient
from src.agent.ports.llm_tooling import LLMAgentTurnResult, LLMToolCall
from src.agent.ports.message_repository import MessageRepository
from src.agent.ports.repository_manager import RepositoryManager, TransactionalOperation
from src.agent.ports.user_memory_repository import UserMemoryRepository
from src.shared.domain.value_objects.user_id import UserId


class InMemoryChatRepository(ChatRepository):
    def __init__(self) -> None:
        self.rows: Dict[ChatId, Chat] = {}

    async def save(self, chat: Chat, tx: Any = None) -> Chat:
        self.rows[chat.chat_id] = chat
        return chat

    async def find_by_id(self, chat_id: ChatId) -> Optional[Chat]:
        return self.rows.get(chat_id)

    async def find_by_user_id(self, user_id: UserId) -> List[Chat]:
        owned = [chat for chat in self.rows.values() if chat.user_id == user_id]
        return sorted(owned, key=lambda chat: chat.created_at, reverse=True)


class InMemoryMessageRepository(MessageRepository):
    def __init__(self) -> None:
        self.rows: Dict[MessageId, Message] = {}

    def _of_chat(self, chat_id: ChatId) -> List[Message]:
        messages = [message for message in self.rows.values() if message.chat_id == chat_id]
        return sorted(messages, key=lambda message: message.created_at)

    async def save(self, message: Message, tx: Any = None) -> Message:
        self.rows[message.message_id] = message
        return message

    async def find_by_chat_id(
        self,
        chat_id: ChatId,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        tx: Any = None,
    ) -> List[Message]:
        messages = self._of_chat(chat_id)[offset or 0:]
        return messages if limit is None else messages[:limit]

    async def find_recent_by_chat_id(self, chat_id: ChatId, limit: int) -> List[Message]:
        return self._of_chat(chat_id)[-limit:]

    async def count_by_chat_id(self, chat_id: ChatId) -> int:
        return len(self._of_chat(chat_id))

    async def update_status(
        self, message_id: MessageId, status: MessageStatus, tx: Any = None
    ) -> None:
        self.rows[message_id].status = status


class InMemoryUserMemoryRepository(UserMemoryRepository):
    def __init__(self) -> None:
        self.rows: Dict[str, UserMemory] = {}

    def _owned(self, user_id: UserId) -> List[UserMemory]:
        owned = [memory for memory in self.rows.values() if memory.user_id == user_id]
        return sorted(owned, key=lambda memory: memory.updated_at, reverse=True)

    async def save(self, memory: UserMemory) -> UserMemory:
        self.rows[memory.memory_id] = memory
        return memory

    async def find_by_user_id(self, user_id: UserId, *, limit: int) -> List[UserMemory]:
        return self._owned(user_id)[:limit]

    async def search_by_user_id(
        self, user_id: UserId, *, query: str, limit: int
    ) -> List[UserMemory]:
        needle = query.lower()
        found = [
            memory
            for memory in self._owned(user_id)
            if needle in memory.content.lower() or needle in (memory.topic or "").lower()
        ]
        return found[:limit]

    async def delete(self, memory_id: str, user_id: UserId) -> None:
        memory = self.rows.get(memory_id)
        if memory is not None and memory.user_id == user_id:
            del self.rows[memory_id]

    async def delete_all_by_user_id(self, user_id: UserId) -> None:
        for memory in self._owned(user_id):
            del self.rows[memory.memory_id]

    async def count_by_user_id(self, user_id: UserId) -> int:
        return len(self._owned(user_id))

    async def find_oldest_by_user_id(self, user_id: UserId) -> Optional[UserMemory]:
        owned = sorted(self._owned(user_id), key=lambda memory: memory.created_at)
        return owned[0] if owned else None


class InMemoryChatAgentContextRepository(ChatAgentContextRepository):
    def __init__(self) -> None:
        self.rows: Dict[ChatId, ChatAgentContext] = {}

    async def save(self, context: ChatAgentContext, tx: Any = None) -> ChatAgentContext:
        self.rows[context.chat_id] = context
        return context

    async def find_by_chat_id(self, chat_id: ChatId) -> Optional[ChatAgentContext]:
        return self.rows.get(chat_id)


class InMemoryAgentRepositoryManager(RepositoryManager):
    def __init__(self) -> None:
        self._chats = InMemoryChatRepository()
        self._messages = InMemoryMessageRepository()
        self._user_memory = InMemoryUserMemoryRepository()
        self._chat_agent_context = InMemoryChatAgentContextRepository()

    @property
    def chats(self) -> InMemoryChatRepository:
        return self._chats

    @property
    def messages(self) -> InMemoryMessageRepository:
        return self._messages

    @property
    def user_memory(self) -> InMemoryUserMemoryRepository:
        return self._user_memory

    @property
    def chat_agent_context(self) -> InMemoryChatAgentContextRepository:
        return self._chat_agent_context

    async def execute_in_transaction(
        self, operations: Sequence[TransactionalOperation]
    ) -> List[Any]:
        return [await operation(None) for operation in operations]


class ScriptedLLMClient(LLMClient):
    """Plays back prepared turns; an exception in the script is raised instead."""

    def __init__(self) -> None:
        self.script: List[Any] = []
        self.calls = 0

    async def run_agent_turn(self, assistant, messages, tools) -> LLMAgentTurnResult:
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def tool_turn(name: str, arguments: Dict[str, Any]) -> LLMAgentTurnResult:
    return LLMAgentTurnResult(
        text="",
        tool_calls=[LLMToolCall(name=name, arguments=arguments, call_id="call-1")],
        tokens=5,
        response_event={"scripted_tool_call": name},
    )


def text_turn(text: str) -> LLMAgentTurnResult:
    return LLMAgentTurnResult(text=text, tokens=3, response_event={"scripted_text": text})
