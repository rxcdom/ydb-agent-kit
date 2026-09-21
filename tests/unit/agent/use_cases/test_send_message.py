"""Tests for the send-message use case: the turn sequence and its failure paths."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, List
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from src.agent.application.context.conversation_state_writer import ConversationStateWriter
from src.agent.application.loop.agent_tool_loop_service import AgentToolLoopService
from src.agent.application.use_cases.send_message import (
    MEMORY_POINTER_TOPIC_SCAN_LIMIT,
    MESSAGE_CONTENT_MAX_LENGTH,
    SendMessageUseCase,
    trace_to_dict,
)
from src.agent.domain.entities.assistant import Assistant
from src.agent.domain.entities.chat import Chat
from src.agent.domain.entities.chat_agent_context import ChatAgentContext
from src.agent.domain.entities.message import Message, MessageRole
from src.agent.domain.entities.user_memory import UserMemory
from src.agent.domain.exceptions import (
    ChatAccessDeniedError,
    ChatNotFoundError,
    InvalidMessageContentError,
    LLMResponseGenerationFailedError,
    LLMServiceTimeoutError,
)
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.domain.value_objects.message_status import MessageStatus
from src.agent.ports.llm_tooling import (
    AgentFunctionCallTrace,
    AgentGenerateResponseResult,
    AgentIterationTrace,
    AgentRequestDebugTrace,
)
from src.shared.domain.exceptions import PersistenceError
from src.shared.domain.value_objects.user_id import UserId

USE_CASE_LOGGER = "src.agent.application.use_cases.send_message"
START = datetime(2026, 9, 17, 12, 5, tzinfo=timezone.utc)
HISTORY_LIMIT = 20

TRACE = AgentRequestDebugTrace(
    model_name="test-model",
    total_tokens=42,
    request_flow=[
        AgentIterationTrace(
            iteration=1,
            tokens=30,
            assistant_text="",
            function_calls=[
                AgentFunctionCallTrace(
                    name="query_tasks",
                    arguments={"statuses": ["open"]},
                    result={
                        "status": "ok",
                        "data": {
                            "window_used": {"from": "2026-01-05", "to": "2026-09-17"},
                            "date_field": "created",
                            "total_count": 3,
                        },
                    },
                    call_id="call-1",
                )
            ],
        ),
        AgentIterationTrace(
            iteration=2, tokens=12, assistant_text="You have 3 open tasks.", function_calls=[]
        ),
    ],
)

EXPECTED_TRACE_DICT = {
    "model_name": "test-model",
    "total_tokens": 42,
    "request_flow": [
        {
            "iteration": 1,
            "tokens": 30,
            "assistant_text": "",
            "function_calls": [
                {
                    "name": "query_tasks",
                    "arguments": {"statuses": ["open"]},
                    "result": {
                        "status": "ok",
                        "data": {
                            "window_used": {"from": "2026-01-05", "to": "2026-09-17"},
                            "date_field": "created",
                            "total_count": 3,
                        },
                    },
                    "call_id": "call-1",
                }
            ],
        },
        {
            "iteration": 2,
            "tokens": 12,
            "assistant_text": "You have 3 open tasks.",
            "function_calls": [],
        },
    ],
}


def _ticking_clock() -> Callable[[], datetime]:
    """A clock that moves one second forward on every reading."""
    readings = iter(START + timedelta(seconds=offset) for offset in range(100))
    return lambda: next(readings)


@pytest.fixture
def assistant() -> Assistant:
    return Assistant(
        name="Task assistant",
        system_prompt="You are a task assistant.",
        model_name="test-model",
        temperature=0.2,
    )


@pytest.fixture
def user_id() -> UserId:
    return UserId.generate()


@pytest.fixture
def chat(user_id, mock_repository_manager) -> Chat:
    created = START - timedelta(days=3)
    owned = Chat(chat_id=ChatId.generate(), user_id=user_id, created_at=created, updated_at=created)
    mock_repository_manager.chats.find_by_id.return_value = owned
    return owned


@pytest.fixture
def loop() -> AsyncMock:
    service = AsyncMock(spec=AgentToolLoopService)
    service.generate_response.return_value = AgentGenerateResponseResult(
        text="You have 3 open tasks.", tokens=42, cost=0.0, debug=TRACE
    )
    return service


@pytest.fixture
def state_writer() -> AsyncMock:
    return AsyncMock(spec=ConversationStateWriter)


@pytest.fixture
def saved_messages(mock_repository_manager) -> List[dict]:
    """Snapshots of every ``messages.save`` call, taken at the time of the call."""
    snapshots: List[dict] = []

    async def record(message: Message, tx: Any = None) -> Message:
        snapshots.append({"message": message, "status": message.status, "tx": tx})
        return message

    mock_repository_manager.messages.save.side_effect = record
    return snapshots


@pytest.fixture
def use_case(mock_repository_manager, loop, state_writer, assistant) -> SendMessageUseCase:
    return SendMessageUseCase(
        repository_manager=mock_repository_manager,
        agent_tool_loop_service=loop,
        conversation_state_writer=state_writer,
        assistant=assistant,
        timezone=ZoneInfo("UTC"),
        history_limit=HISTORY_LIMIT,
        clock=_ticking_clock(),
    )


async def test_successful_turn_stores_both_messages_and_returns_the_trace(
    use_case, mock_repository_manager, transaction, saved_messages, state_writer, chat, user_id
):
    result = await use_case.execute(user_id, chat.chat_id, "  What is open?  ")

    # Step 2: the user message is stored as processing, outside any transaction.
    first = saved_messages[0]
    assert first["message"] is result.user_message
    assert first["status"] == MessageStatus.PROCESSING
    assert first["tx"] is None
    assert result.user_message.role == MessageRole.USER
    assert result.user_message.content == "What is open?"
    assert result.user_message.user_id == user_id
    assert result.user_message.chat_id == chat.chat_id
    assert result.user_message.tokens == 0
    assert result.user_message.trace is None

    # Step 7: one transaction flips the user message and stores the reply.
    mock_repository_manager.execute_in_transaction.assert_awaited_once()
    assert len(mock_repository_manager.execute_in_transaction.await_args.args[0]) == 3
    mock_repository_manager.messages.update_status.assert_awaited_once_with(
        result.user_message.message_id, MessageStatus.SENT, tx=transaction
    )
    assert len(saved_messages) == 2
    second = saved_messages[1]
    assert second["message"] is result.assistant_message
    assert second["status"] == MessageStatus.SENT
    assert second["tx"] is transaction
    mock_repository_manager.chats.save.assert_awaited_once_with(chat, tx=transaction)

    assistant_message = result.assistant_message
    assert assistant_message.role == MessageRole.ASSISTANT
    assert assistant_message.content == "You have 3 open tasks."
    assert assistant_message.tokens == 42
    assert assistant_message.status == MessageStatus.SENT
    assert assistant_message.user_id == user_id
    assert assistant_message.chat_id == chat.chat_id
    assert assistant_message.message_id != result.user_message.message_id
    assert assistant_message.created_at > result.user_message.created_at
    assert assistant_message.trace == EXPECTED_TRACE_DICT
    assert json.loads(json.dumps(assistant_message.trace)) == EXPECTED_TRACE_DICT

    # The chat is touched, and what is returned shows the committed status.
    assert chat.updated_at == assistant_message.created_at
    assert result.user_message.status == MessageStatus.SENT
    assert result.debug is TRACE

    # Step 8 happens after the commit, with the trace of this turn.
    state_writer.record_turn.assert_awaited_once_with(
        chat_id=chat.chat_id, user_id=user_id, trace=TRACE
    )


async def test_loop_receives_the_history_the_blocks_and_the_owner(
    use_case, mock_repository_manager, loop, assistant, chat, user_id
):
    earlier = Message(
        message_id=result_id(),
        chat_id=chat.chat_id,
        user_id=user_id,
        role=MessageRole.ASSISTANT,
        content="Hello!",
        tokens=5,
        status=MessageStatus.SENT,
        created_at=START - timedelta(minutes=5),
    )
    mock_repository_manager.messages.find_recent_by_chat_id.return_value = [earlier]
    context = ChatAgentContext.empty_for(chat.chat_id, user_id)
    context.record_windowed_tool(
        tool="query_tasks",
        window_from="2026-09-01",
        window_to="2026-09-17",
        date_field="due",
        project="Kitchen",
    )
    mock_repository_manager.chat_agent_context.find_by_chat_id.return_value = context
    mock_repository_manager.user_memory.count_by_user_id.return_value = 2
    mock_repository_manager.user_memory.find_by_user_id.return_value = [
        UserMemory(
            memory_id=UserMemory.generate_id(),
            user_id=user_id,
            content="Reviews open tasks every Friday",
            topic="routine",
            created_at=START,
            updated_at=START,
        )
    ]

    await use_case.execute(user_id, chat.chat_id, "What is due?")

    mock_repository_manager.messages.find_recent_by_chat_id.assert_awaited_once_with(
        chat.chat_id, HISTORY_LIMIT
    )
    mock_repository_manager.user_memory.find_by_user_id.assert_awaited_once_with(
        user_id, limit=MEMORY_POINTER_TOPIC_SCAN_LIMIT
    )
    call = loop.generate_response.await_args
    assert call.args == (assistant, [earlier], "What is due?")
    assert set(call.kwargs) == {
        "owner_user_id",
        "calendar_block",
        "conversation_state_block",
        "memory_pointer",
    }
    assert call.kwargs["owner_user_id"] == str(user_id)
    assert call.kwargs["calendar_block"].startswith("## Calendar\nNow: 2026-09-17 (Thursday)")
    assert call.kwargs["conversation_state_block"].startswith("## Conversation state")
    assert "- last_project: Kitchen" in call.kwargs["conversation_state_block"]
    assert call.kwargs["memory_pointer"].startswith("## Long-term memory")
    assert "Entries: 2." in call.kwargs["memory_pointer"]
    assert "Topics: routine." in call.kwargs["memory_pointer"]
    # The pointer names topics only; what was remembered stays in the vault.
    assert "Friday" not in call.kwargs["memory_pointer"]


def result_id():
    from src.agent.domain.value_objects.message_id import MessageId

    return MessageId.generate()


async def test_history_is_loaded_after_the_user_message_is_stored(
    use_case, mock_repository_manager, chat, user_id
):
    order: List[str] = []
    mock_repository_manager.messages.save.side_effect = (
        lambda message, tx=None: order.append(f"save:{message.role.value}")
    )
    mock_repository_manager.messages.find_recent_by_chat_id.side_effect = (
        lambda chat_id, limit: order.append("load") or []
    )

    await use_case.execute(user_id, chat.chat_id, "What is open?")

    assert order == ["save:user", "load", "save:assistant"]


async def test_calendar_block_is_rendered_in_the_configured_time_zone(
    mock_repository_manager, loop, state_writer, assistant, chat, user_id
):
    late_evening_utc = datetime(2026, 9, 17, 23, 30, tzinfo=timezone.utc)
    use_case = SendMessageUseCase(
        repository_manager=mock_repository_manager,
        agent_tool_loop_service=loop,
        conversation_state_writer=state_writer,
        assistant=assistant,
        timezone=ZoneInfo("Asia/Tokyo"),
        history_limit=HISTORY_LIMIT,
        clock=lambda: late_evening_utc,
    )

    await use_case.execute(user_id, chat.chat_id, "What is overdue?")

    calendar_block = loop.generate_response.await_args.kwargs["calendar_block"]
    assert "Now: 2026-09-18 (Friday), 08:30 in Asia/Tokyo." in calendar_block
    assert "- yesterday: 2026-09-17" in calendar_block


@pytest.mark.parametrize("content", ["", "   ", "\n\t", "x" * (MESSAGE_CONTENT_MAX_LENGTH + 1)])
async def test_unacceptable_content_is_rejected_before_anything_is_stored(
    use_case, mock_repository_manager, loop, chat, user_id, content
):
    with pytest.raises(InvalidMessageContentError):
        await use_case.execute(user_id, chat.chat_id, content)

    mock_repository_manager.chats.find_by_id.assert_not_awaited()
    mock_repository_manager.messages.save.assert_not_awaited()
    loop.generate_response.assert_not_awaited()


async def test_content_of_exactly_the_maximum_length_is_accepted(use_case, chat, user_id):
    result = await use_case.execute(user_id, chat.chat_id, "x" * MESSAGE_CONTENT_MAX_LENGTH)

    assert len(result.user_message.content) == MESSAGE_CONTENT_MAX_LENGTH


async def test_missing_chat_is_not_found(use_case, mock_repository_manager, loop, user_id):
    mock_repository_manager.chats.find_by_id.return_value = None

    with pytest.raises(ChatNotFoundError):
        await use_case.execute(user_id, ChatId.generate(), "What is open?")

    mock_repository_manager.messages.save.assert_not_awaited()
    loop.generate_response.assert_not_awaited()


async def test_chat_of_another_owner_is_denied_and_left_untouched(
    use_case, mock_repository_manager, loop, state_writer, chat
):
    intruder = UserId.generate()

    with pytest.raises(ChatAccessDeniedError):
        await use_case.execute(intruder, chat.chat_id, "What is open?")

    mock_repository_manager.messages.save.assert_not_awaited()
    mock_repository_manager.messages.find_recent_by_chat_id.assert_not_awaited()
    mock_repository_manager.user_memory.count_by_user_id.assert_not_awaited()
    mock_repository_manager.execute_in_transaction.assert_not_awaited()
    loop.generate_response.assert_not_awaited()
    state_writer.record_turn.assert_not_awaited()


async def test_llm_failure_marks_the_user_message_failed_and_names_it(
    use_case, mock_repository_manager, transaction, saved_messages, loop, state_writer, chat,
    user_id,
):
    failure = LLMServiceTimeoutError("the provider did not answer in time")
    loop.generate_response.side_effect = failure

    with pytest.raises(LLMResponseGenerationFailedError) as raised:
        await use_case.execute(user_id, chat.chat_id, "What is open?")

    user_message = saved_messages[0]["message"]
    assert raised.value.user_message_id == str(user_message.message_id)
    assert raised.value.cause is failure
    assert raised.value.__cause__ is failure

    # The terminal status is written once, in one transaction, and it is failed.
    mock_repository_manager.execute_in_transaction.assert_awaited_once()
    assert len(mock_repository_manager.execute_in_transaction.await_args.args[0]) == 1
    mock_repository_manager.messages.update_status.assert_awaited_once_with(
        user_message.message_id, MessageStatus.FAILED, tx=transaction
    )
    assert user_message.status == MessageStatus.FAILED

    # No reply exists, the chat is not touched, and no state is recorded.
    assert len(saved_messages) == 1
    mock_repository_manager.chats.save.assert_not_awaited()
    state_writer.record_turn.assert_not_awaited()


async def test_unreadable_memory_vault_downgrades_to_no_pointer(
    use_case, mock_repository_manager, loop, chat, user_id, caplog
):
    mock_repository_manager.user_memory.count_by_user_id.side_effect = PersistenceError("down")

    with caplog.at_level(logging.WARNING, logger=USE_CASE_LOGGER):
        result = await use_case.execute(user_id, chat.chat_id, "What is open?")

    assert result.assistant_message.content == "You have 3 open tasks."
    assert loop.generate_response.await_args.kwargs["memory_pointer"] == ""
    records = [r for r in caplog.records if "Memory pointer not rendered" in r.getMessage()]
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert records[0].exc_info[0] is PersistenceError


async def test_unreadable_conversation_state_downgrades_to_no_block(
    use_case, mock_repository_manager, loop, chat, user_id, caplog
):
    mock_repository_manager.chat_agent_context.find_by_chat_id.side_effect = PersistenceError(
        "down"
    )

    with caplog.at_level(logging.WARNING, logger=USE_CASE_LOGGER):
        result = await use_case.execute(user_id, chat.chat_id, "What is open?")

    assert result.user_message.status == MessageStatus.SENT
    assert loop.generate_response.await_args.kwargs["conversation_state_block"] == ""
    records = [r for r in caplog.records if "Conversation state not rendered" in r.getMessage()]
    assert len(records) == 1
    assert records[0].exc_info is not None


async def test_failed_state_capture_does_not_fail_the_turn(
    use_case, mock_repository_manager, state_writer, chat, user_id, caplog
):
    state_writer.record_turn.side_effect = PersistenceError("down")

    with caplog.at_level(logging.WARNING, logger=USE_CASE_LOGGER):
        result = await use_case.execute(user_id, chat.chat_id, "What is open?")

    assert result.assistant_message.status == MessageStatus.SENT
    assert result.user_message.status == MessageStatus.SENT
    mock_repository_manager.execute_in_transaction.assert_awaited_once()
    records = [r for r in caplog.records if "Conversation state not recorded" in r.getMessage()]
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert records[0].exc_info[0] is PersistenceError


async def test_programming_error_in_a_side_path_is_not_swallowed(
    use_case, state_writer, chat, user_id
):
    """Only datastore and other domain failures are tolerated on the side paths."""
    state_writer.record_turn.side_effect = RuntimeError("a bug, not an outage")

    with pytest.raises(RuntimeError):
        await use_case.execute(user_id, chat.chat_id, "What is open?")


async def test_storage_failure_while_committing_is_raised(
    use_case, mock_repository_manager, state_writer, chat, user_id
):
    mock_repository_manager.execute_in_transaction.side_effect = PersistenceError("down")

    with pytest.raises(PersistenceError):
        await use_case.execute(user_id, chat.chat_id, "What is open?")

    state_writer.record_turn.assert_not_awaited()


def test_history_limit_below_one_is_rejected(
    mock_repository_manager, loop, state_writer, assistant
):
    with pytest.raises(ValueError, match="history_limit"):
        SendMessageUseCase(
            repository_manager=mock_repository_manager,
            agent_tool_loop_service=loop,
            conversation_state_writer=state_writer,
            assistant=assistant,
            timezone=ZoneInfo("UTC"),
            history_limit=0,
        )


def test_trace_to_dict_returns_plain_json_data():
    as_dict = trace_to_dict(TRACE)

    assert as_dict == EXPECTED_TRACE_DICT
    # A copy: changing the stored form never reaches the trace the loop returned.
    as_dict["request_flow"][0]["function_calls"][0]["result"]["status"] = "changed"
    assert TRACE.request_flow[0].function_calls[0].result["status"] == "ok"
