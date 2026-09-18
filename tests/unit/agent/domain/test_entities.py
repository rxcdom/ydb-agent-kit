"""Tests for the agent entities, value objects and exception hierarchy."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.agent.domain.entities.assistant import Assistant
from src.agent.domain.entities.chat import CHAT_TITLE_MAX_LENGTH, Chat
from src.agent.domain.entities.message import Message, MessageRole
from src.agent.domain.entities.user_memory import (
    USER_MEMORY_CONTENT_MAX_LENGTH,
    USER_MEMORY_TOPIC_MAX_LENGTH,
    UserMemory,
)
from src.agent.domain.exceptions import (
    ChatAccessDeniedError,
    ChatNotFoundError,
    InvalidMessageContentError,
    LLMResponseGenerationFailedError,
    LLMServiceAuthenticationError,
    LLMServiceError,
    LLMServiceInvalidResponseError,
    LLMServiceTimeoutError,
    LLMServiceUnavailableError,
)
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.domain.value_objects.message_id import MessageId
from src.agent.domain.value_objects.message_status import MessageStatus
from src.shared.domain.exceptions import (
    AccessDeniedError,
    DomainError,
    NotFoundError,
    ValidationError,
)
from src.shared.domain.value_objects.user_id import UserId


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _message(**overrides) -> Message:
    fields = dict(
        message_id=MessageId.generate(),
        chat_id=ChatId.generate(),
        user_id=UserId.generate(),
        role=MessageRole.USER,
        content="What is overdue this week?",
        tokens=0,
        status=MessageStatus.PROCESSING,
        created_at=_now(),
    )
    fields.update(overrides)
    return Message(**fields)


def _memory(**overrides) -> UserMemory:
    now = _now()
    fields = dict(
        memory_id=UserMemory.generate_id(),
        user_id=UserId.generate(),
        content="Reviews open tasks every Friday",
        topic="routine",
        created_at=now,
        updated_at=now,
    )
    fields.update(overrides)
    return UserMemory(**fields)


def test_chat_creation():
    user_id = UserId.generate()
    chat_id = ChatId.generate()
    now = _now()

    chat = Chat(chat_id=chat_id, user_id=user_id, created_at=now, updated_at=now)

    assert chat.chat_id == chat_id
    assert chat.user_id == user_id
    assert chat.title is None


def test_chat_title_is_trimmed_and_blank_becomes_none():
    now = _now()
    titled = Chat(ChatId.generate(), UserId.generate(), now, now, title="  Weekly review ")
    blank = Chat(ChatId.generate(), UserId.generate(), now, now, title="   ")

    assert titled.title == "Weekly review"
    assert blank.title is None


def test_chat_title_length_is_capped():
    now = _now()
    with pytest.raises(ValueError, match="Chat title"):
        Chat(
            ChatId.generate(),
            UserId.generate(),
            now,
            now,
            title="x" * (CHAT_TITLE_MAX_LENGTH + 1),
        )


def test_message_creation():
    message_id = MessageId.generate()
    chat_id = ChatId.generate()
    user_id = UserId.generate()

    message = _message(message_id=message_id, chat_id=chat_id, user_id=user_id)

    assert message.message_id == message_id
    assert message.chat_id == chat_id
    assert message.user_id == user_id
    assert message.role is MessageRole.USER
    assert message.content == "What is overdue this week?"
    assert message.tokens == 0
    assert message.status is MessageStatus.PROCESSING
    assert message.trace is None


def test_assistant_message_carries_its_trace():
    trace = {"model_name": "demo-model", "total_tokens": 42, "request_flow": []}
    message = _message(
        role=MessageRole.ASSISTANT,
        content="Two tasks are overdue.",
        tokens=42,
        status=MessageStatus.SENT,
        trace=trace,
    )

    assert message.role is MessageRole.ASSISTANT
    assert message.trace == trace


def test_message_validation():
    with pytest.raises(ValueError, match="Message content cannot be empty"):
        _message(content="")
    with pytest.raises(ValueError, match="Message content cannot be empty"):
        _message(content="   ")
    with pytest.raises(ValueError, match="Tokens cannot be negative"):
        _message(tokens=-1)


def test_role_and_status_serialise_as_plain_strings():
    assert MessageRole("assistant") is MessageRole.ASSISTANT
    assert MessageRole.USER.value == "user"
    assert [status.value for status in MessageStatus] == ["processing", "sent", "failed"]


def test_identifiers_round_trip_through_strings():
    chat_id = ChatId.generate()
    message_id = MessageId.generate()

    assert ChatId.from_string(str(chat_id)) == chat_id
    assert MessageId.from_string(str(message_id)) == message_id
    with pytest.raises(ValueError):
        ChatId.from_string("not-a-uuid")


def test_assistant_invariants():
    assistant = Assistant(
        name="Task assistant",
        system_prompt="Answer from the tools only.",
        model_name="demo-model",
        temperature=0.2,
    )
    assert assistant.temperature == 0.2

    valid = dict(
        name="Task assistant",
        system_prompt="Answer from the tools only.",
        model_name="demo-model",
        temperature=0.2,
    )
    for field_name in ("name", "system_prompt", "model_name"):
        with pytest.raises(ValueError, match="cannot be empty"):
            Assistant(**{**valid, field_name: "  "})
    for temperature in (-0.1, 2.1):
        with pytest.raises(ValueError, match="Temperature"):
            Assistant(**{**valid, "temperature": temperature})


def test_user_memory_invariants():
    memory = _memory()
    assert memory.topic == "routine"
    assert _memory(topic="   ").topic is None
    assert _memory(topic=None).topic is None
    assert UserMemory.generate_id() != UserMemory.generate_id()

    with pytest.raises(ValueError, match="memory_id"):
        _memory(memory_id=" ")
    with pytest.raises(ValueError, match="content must be non-empty"):
        _memory(content="  ")
    with pytest.raises(ValueError, match="content must be at most"):
        _memory(content="x" * (USER_MEMORY_CONTENT_MAX_LENGTH + 1))
    with pytest.raises(ValueError, match="topic must be at most"):
        _memory(topic="x" * (USER_MEMORY_TOPIC_MAX_LENGTH + 1))


def test_exception_hierarchy_maps_onto_the_shared_base_classes():
    for llm_error in (
        LLMServiceUnavailableError,
        LLMServiceTimeoutError,
        LLMServiceAuthenticationError,
        LLMServiceInvalidResponseError,
        LLMResponseGenerationFailedError,
    ):
        assert issubclass(llm_error, LLMServiceError)
    assert issubclass(LLMServiceError, DomainError)
    assert issubclass(ChatNotFoundError, NotFoundError)
    assert issubclass(ChatAccessDeniedError, AccessDeniedError)
    assert issubclass(InvalidMessageContentError, ValidationError)


def test_llm_errors_keep_the_original_cause():
    original = ConnectionError("socket closed")

    explicit = LLMServiceUnavailableError("provider unreachable", cause=original)
    assert explicit.cause is original
    assert explicit.__cause__ is original
    assert str(explicit) == "provider unreachable"

    with pytest.raises(LLMServiceTimeoutError) as raised:
        try:
            raise TimeoutError("deadline exceeded")
        except TimeoutError as error:
            raise LLMServiceTimeoutError("provider timed out") from error
    assert isinstance(raised.value.cause, TimeoutError)

    assert LLMServiceError("no cause").cause is None


def test_generation_failed_error_names_the_persisted_user_message():
    message_id = str(MessageId.generate())
    cause = LLMServiceUnavailableError("provider unreachable")

    error = LLMResponseGenerationFailedError(
        "The reply could not be generated", user_message_id=message_id, cause=cause
    )

    assert error.user_message_id == message_id
    assert error.cause is cause
    with pytest.raises(TypeError):
        LLMResponseGenerationFailedError("missing the message id")  # type: ignore[call-arg]
