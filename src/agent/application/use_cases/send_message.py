"""Send one user message to a chat and store the assistant's reply.

The turn, step by step:

1. check that the chat exists and belongs to the principal;
2. store the user message as ``processing``, so it survives whatever follows;
3. load the recent history of the chat;
4. render the context blocks of the system message;
5. run the agent loop;
6. on a model failure mark the user message ``failed`` and report its id, so
   the client can show the failure on that message and retry;
7. otherwise commit the outcome in one transaction: the user message becomes
   ``sent``, the assistant message is stored with its usage and trace, and the
   chat is touched;
8. record the conversation state for the next turn.

The terminal status of the user message is written exactly once, by step 6 or
by step 7, never by both. Rendering the context blocks and recording the state
are conveniences: when the datastore fails there, the failure is logged and the
turn goes on without them.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Dict, List
from zoneinfo import ZoneInfo

from src.agent.application.context.conversation_state_writer import ConversationStateWriter
from src.agent.application.loop.agent_tool_loop_service import AgentToolLoopService
from src.agent.domain.entities.assistant import Assistant
from src.agent.domain.entities.chat import Chat
from src.agent.domain.entities.message import Message, MessageRole
from src.agent.domain.exceptions import (
    ChatAccessDeniedError,
    ChatNotFoundError,
    InvalidMessageContentError,
    LLMResponseGenerationFailedError,
    LLMServiceError,
)
from src.agent.domain.services.calendar_block_renderer import CalendarBlockRenderer
from src.agent.domain.services.conversation_state_renderer import ConversationStateRenderer
from src.agent.domain.services.memory_pointer_renderer import MemoryPointerRenderer
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.domain.value_objects.message_id import MessageId
from src.agent.domain.value_objects.message_status import MessageStatus
from src.agent.ports.llm_tooling import AgentGenerateResponseResult, AgentRequestDebugTrace
from src.agent.ports.repository_manager import RepositoryManager
from src.shared.domain.exceptions import DomainError
from src.shared.domain.value_objects.user_id import UserId

logger = logging.getLogger(__name__)

MESSAGE_CONTENT_MAX_LENGTH = 10000

# How many of the freshest vault entries are scanned for topics when the memory
# pointer is rendered. The renderer caps what it shows on its own.
MEMORY_POINTER_TOPIC_SCAN_LIMIT = 30


def _utc_now() -> datetime:
    return datetime.now(UTC)


def trace_to_dict(trace: AgentRequestDebugTrace) -> Dict[str, Any]:
    """Return the trace as plain dictionaries and lists, ready to be stored as JSON.

    Shape::

        {"model_name": str, "total_tokens": int, "request_flow": [
            {"iteration": int, "tokens": int, "assistant_text": str, "function_calls": [
                {"name": str, "arguments": {...}, "result": {...}, "call_id": str | None}]}]}
    """
    return asdict(trace)


@dataclass(frozen=True)
class SendMessageResult:
    """Both messages of the turn and the trace of how the reply was produced.

    ``assistant_message.trace`` holds the same trace as a plain dictionary.
    """

    user_message: Message
    assistant_message: Message
    debug: AgentRequestDebugTrace


class SendMessageUseCase:
    """Run one turn of a chat for the authenticated user."""

    def __init__(
        self,
        repository_manager: RepositoryManager,
        agent_tool_loop_service: AgentToolLoopService,
        conversation_state_writer: ConversationStateWriter,
        assistant: Assistant,
        timezone: ZoneInfo,
        history_limit: int,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be at least 1")

        self._repositories = repository_manager
        self._loop = agent_tool_loop_service
        self._conversation_state_writer = conversation_state_writer
        self._assistant = assistant
        self._timezone = timezone
        self._history_limit = history_limit
        self._clock = clock

    async def execute(self, user_id: UserId, chat_id: ChatId, content: str) -> SendMessageResult:
        """Answer ``content`` in the chat and return the stored messages.

        Raises ``InvalidMessageContentError`` for blank or oversized content,
        ``ChatNotFoundError`` for a missing chat, ``ChatAccessDeniedError`` for a
        chat of another owner, and ``LLMResponseGenerationFailedError`` when the
        model could not answer; the user message is then stored as ``failed``.
        """
        text = self._validated_content(content)
        chat = await self._load_owned_chat(chat_id, user_id)

        now = self._clock()
        user_message = Message(
            message_id=MessageId.generate(),
            chat_id=chat_id,
            user_id=user_id,
            role=MessageRole.USER,
            content=text,
            tokens=0,
            status=MessageStatus.PROCESSING,
            created_at=now,
        )
        await self._repositories.messages.save(user_message)

        history = await self._repositories.messages.find_recent_by_chat_id(
            chat_id, self._history_limit
        )
        calendar_block = CalendarBlockRenderer.render(now, self._timezone)
        conversation_state_block = await self._render_conversation_state(chat_id)
        memory_pointer = await self._render_memory_pointer(user_id, chat_id)

        try:
            generation = await self._loop.generate_response(
                self._assistant,
                history,
                text,
                owner_user_id=str(user_id),
                calendar_block=calendar_block,
                conversation_state_block=conversation_state_block,
                memory_pointer=memory_pointer,
            )
        except LLMServiceError as error:
            await self._mark_failed(user_message)
            logger.warning(
                "Turn failed in chat %s: user message %s marked failed (%s)",
                chat_id,
                user_message.message_id,
                type(error).__name__,
            )
            raise LLMResponseGenerationFailedError(
                "The assistant could not answer; the user message is stored as failed",
                user_message_id=str(user_message.message_id),
                cause=error,
            ) from error

        assistant_message = await self._commit_turn(chat, user_message, generation)
        await self._record_conversation_state(chat_id, user_id, generation.debug)

        logger.info(
            "Turn completed in chat %s: iterations=%d tokens=%d",
            chat_id,
            len(generation.debug.request_flow),
            generation.tokens,
        )
        return SendMessageResult(
            user_message=user_message,
            assistant_message=assistant_message,
            debug=generation.debug,
        )

    @staticmethod
    def _validated_content(content: str) -> str:
        text = (content or "").strip()
        if not text:
            raise InvalidMessageContentError("Message content cannot be empty")
        if len(text) > MESSAGE_CONTENT_MAX_LENGTH:
            raise InvalidMessageContentError(
                f"Message content exceeds {MESSAGE_CONTENT_MAX_LENGTH} characters"
            )
        return text

    async def _load_owned_chat(self, chat_id: ChatId, user_id: UserId) -> Chat:
        chat = await self._repositories.chats.find_by_id(chat_id)
        if chat is None:
            raise ChatNotFoundError(f"Chat {chat_id} not found")
        if chat.user_id != user_id:
            raise ChatAccessDeniedError(f"Chat {chat_id} belongs to another user")
        return chat

    async def _render_conversation_state(self, chat_id: ChatId) -> str:
        """Render the conversation-state block; an unreadable state means no block."""
        try:
            context = await self._repositories.chat_agent_context.find_by_chat_id(chat_id)
        except DomainError:
            logger.warning(
                "Conversation state not rendered for chat %s", chat_id, exc_info=True
            )
            return ""
        return ConversationStateRenderer.render(context)

    async def _render_memory_pointer(self, user_id: UserId, chat_id: ChatId) -> str:
        """Render the memory pointer; an unreadable vault means no pointer."""
        try:
            count = await self._repositories.user_memory.count_by_user_id(user_id)
            recent = await self._repositories.user_memory.find_by_user_id(
                user_id, limit=MEMORY_POINTER_TOPIC_SCAN_LIMIT
            )
        except DomainError:
            logger.warning("Memory pointer not rendered for chat %s", chat_id, exc_info=True)
            return ""
        return MemoryPointerRenderer.render(count, [memory.topic for memory in recent])

    async def _mark_failed(self, user_message: Message) -> None:
        async def mark_user_message_failed(tx: Any) -> None:
            await self._repositories.messages.update_status(
                user_message.message_id, MessageStatus.FAILED, tx=tx
            )

        await self._repositories.execute_in_transaction([mark_user_message_failed])
        user_message.status = MessageStatus.FAILED

    async def _commit_turn(
        self, chat: Chat, user_message: Message, generation: AgentGenerateResponseResult
    ) -> Message:
        """Store the outcome of a successful turn atomically.

        Either the reply exists and the user message is ``sent``, or neither
        happened. Every operation writes a fixed row, so a replayed transaction
        ends in the same state.
        """
        assistant_message = Message(
            message_id=MessageId.generate(),
            chat_id=chat.chat_id,
            user_id=chat.user_id,
            role=MessageRole.ASSISTANT,
            content=generation.text,
            tokens=generation.tokens,
            status=MessageStatus.SENT,
            created_at=self._clock(),
            trace=trace_to_dict(generation.debug),
        )
        chat.updated_at = assistant_message.created_at

        async def mark_user_message_sent(tx: Any) -> None:
            await self._repositories.messages.update_status(
                user_message.message_id, MessageStatus.SENT, tx=tx
            )

        async def save_assistant_message(tx: Any) -> Message:
            return await self._repositories.messages.save(assistant_message, tx=tx)

        async def touch_chat(tx: Any) -> Chat:
            return await self._repositories.chats.save(chat, tx=tx)

        operations: List[Callable[[Any], Any]] = [
            mark_user_message_sent,
            save_assistant_message,
            touch_chat,
        ]
        await self._repositories.execute_in_transaction(operations)
        user_message.status = MessageStatus.SENT
        return assistant_message

    async def _record_conversation_state(
        self, chat_id: ChatId, user_id: UserId, trace: AgentRequestDebugTrace
    ) -> None:
        """Record the state for the next turn; the reply is already committed."""
        try:
            await self._conversation_state_writer.record_turn(
                chat_id=chat_id, user_id=user_id, trace=trace
            )
        except DomainError:
            logger.warning(
                "Conversation state not recorded for chat %s", chat_id, exc_info=True
            )
