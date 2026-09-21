"""Tests for the agent tool loop: history, window, tool round-trips, trace, fallback."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, List
from unittest.mock import AsyncMock

import pytest

from src.agent.application.loop.agent_tool_loop_service import (
    AGENT_LOOP_FALLBACK_TEXT,
    AgentToolLoopService,
)
from src.agent.application.loop.system_prompt_composer import SystemPromptComposer
from src.agent.application.tools import build_tool_registry
from src.agent.application.tools.dispatcher import ToolDispatcher
from src.agent.domain.entities.assistant import Assistant
from src.agent.domain.entities.message import Message, MessageRole
from src.agent.domain.exceptions import LLMServiceUnavailableError
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.domain.value_objects.message_id import MessageId
from src.agent.domain.value_objects.message_status import MessageStatus
from src.agent.ports.llm_client import LLMClient
from src.agent.ports.llm_tooling import LLMAgentTurnResult, LLMToolCall, LLMToolResult
from src.agent.ports.task_data_provider import TaskDataProvider
from src.shared.domain.value_objects.user_id import UserId

OWNER = "0b5c1f0e-6f0a-4a57-9d55-3f1f4a1c2b10"
PROMPT = "You are a task assistant."
CALENDAR = "## Calendar\nNow: 2026-09-17 (Thursday), 14:05 in UTC."
POINTER = "## Long-term memory\n\nEntries: 1."
BLOCKS = {
    "calendar_block": CALENDAR,
    "conversation_state_block": "",
    "memory_pointer": POINTER,
}


class _ResponseEvent:
    """Stands in for the provider's own result object of one model turn."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return f"<event {self.label}>"


@pytest.fixture
def assistant() -> Assistant:
    return Assistant(
        name="Task assistant", system_prompt=PROMPT, model_name="test-model", temperature=0.2
    )


@pytest.fixture
def task_provider() -> AsyncMock:
    provider = AsyncMock(spec=TaskDataProvider)
    provider.query_tasks.return_value = {"status": "ok", "data": {"total_count": 3}}
    return provider


def _llm(*turns: LLMAgentTurnResult) -> AsyncMock:
    client = AsyncMock(spec=LLMClient)
    client.run_agent_turn.side_effect = list(turns)
    return client


def _service(
    llm: AsyncMock,
    *,
    provider: Any = None,
    dispatcher: Any = None,
    max_iterations: int = 5,
    history_limit: int = 20,
) -> AgentToolLoopService:
    registry = build_tool_registry(provider, None)
    return AgentToolLoopService(
        llm_client=llm,
        tool_dispatcher=dispatcher if dispatcher is not None else ToolDispatcher(registry),
        tool_registry=registry,
        system_prompt_composer=SystemPromptComposer(),
        max_iterations=max_iterations,
        history_limit=history_limit,
    )


def _stored(content: str, role: MessageRole) -> Message:
    return Message(
        message_id=MessageId.generate(),
        chat_id=ChatId.generate(),
        user_id=UserId.from_string(OWNER),
        role=role,
        content=content,
        tokens=0,
        status=MessageStatus.SENT,
        created_at=datetime.now(timezone.utc),
    )


def _text_turn(text: str, tokens: int = 1, label: str = "text") -> LLMAgentTurnResult:
    return LLMAgentTurnResult(
        text=text, tool_calls=[], tokens=tokens, response_event=_ResponseEvent(label)
    )


def _tool_turn(
    *calls: LLMToolCall, tokens: int = 1, label: str = "tools"
) -> LLMAgentTurnResult:
    return LLMAgentTurnResult(
        text="", tool_calls=list(calls), tokens=tokens, response_event=_ResponseEvent(label)
    )


def _sent_messages(llm: AsyncMock, call_index: int = 0) -> List[Any]:
    return llm.run_agent_turn.await_args_list[call_index].kwargs["messages"]


def _is_tool_results(item: Any) -> bool:
    return isinstance(item, dict) and "tool_results" in item


async def test_returns_final_text_without_tool_calls(assistant):
    llm = _llm(_text_turn("  You have 3 open tasks.  ", tokens=12))
    service = _service(llm)

    result = await service.generate_response(
        assistant, [], "What is open?", owner_user_id=OWNER, **BLOCKS
    )

    assert result.text == "You have 3 open tasks."
    assert result.tokens == 12
    assert result.debug.model_name == "test-model"
    assert result.debug.total_tokens == 12
    assert len(result.debug.request_flow) == 1
    assert result.debug.request_flow[0].iteration == 1
    assert result.debug.request_flow[0].function_calls == []
    assert llm.run_agent_turn.await_count == 1

    kwargs = llm.run_agent_turn.await_args.kwargs
    assert kwargs["assistant"] is assistant
    assert kwargs["messages"] == [
        {"role": "system", "text": f"{PROMPT}\n\n{CALENDAR}\n\n{POINTER}"},
        {"role": "user", "text": "What is open?"},
    ]
    assert [tool.name for tool in kwargs["tools"]] == [
        "list_projects",
        "query_tasks",
        "create_task",
        "update_task",
        "delete_task",
        "remember",
        "recall",
        "forget",
    ]


async def test_stored_turns_are_sent_oldest_first_with_their_roles(assistant):
    llm = _llm(_text_turn("Done."))
    history = [
        _stored("What is open?", MessageRole.USER),
        _stored("You have 3 open tasks.", MessageRole.ASSISTANT),
    ]

    await _service(llm).generate_response(
        assistant, history, "And in Kitchen?", owner_user_id=OWNER, **BLOCKS
    )

    assert _sent_messages(llm)[1:] == [
        {"role": "user", "text": "What is open?"},
        {"role": "assistant", "text": "You have 3 open tasks."},
        {"role": "user", "text": "And in Kitchen?"},
    ]


async def test_new_message_already_stored_as_the_last_turn_is_not_repeated(assistant):
    llm = _llm(_text_turn("Done."))
    history = [
        _stored("What is open?", MessageRole.USER),
        _stored("You have 3 open tasks.", MessageRole.ASSISTANT),
        _stored("And in Kitchen?", MessageRole.USER),
    ]

    await _service(llm).generate_response(
        assistant, history, "  And in Kitchen?  ", owner_user_id=OWNER, **BLOCKS
    )

    user_texts = [item["text"] for item in _sent_messages(llm) if item["role"] == "user"]
    assert user_texts == ["What is open?", "And in Kitchen?"]


@pytest.mark.parametrize(
    ("last_role", "last_text"),
    [
        (MessageRole.USER, "What is open?"),
        (MessageRole.ASSISTANT, "And in Kitchen?"),
    ],
    ids=["different-user-text", "same-text-from-assistant"],
)
async def test_new_message_is_appended_unless_the_last_stored_turn_is_that_message(
    assistant, last_role, last_text
):
    llm = _llm(_text_turn("Done."))

    await _service(llm).generate_response(
        assistant, [_stored(last_text, last_role)], "And in Kitchen?", owner_user_id=OWNER, **BLOCKS
    )

    assert _sent_messages(llm)[1:] == [
        {"role": last_role.value, "text": last_text},
        {"role": "user", "text": "And in Kitchen?"},
    ]


async def test_schema_error_is_fed_back_and_the_model_corrects_itself(assistant, task_provider):
    bad_call = LLMToolCall(name="query_tasks", arguments={"unexpected": True}, call_id="call-1")
    good_call = LLMToolCall(name="query_tasks", arguments={"statuses": ["open"]}, call_id="call-2")
    first, second = _tool_turn(bad_call, tokens=4, label="1"), _tool_turn(
        good_call, tokens=5, label="2"
    )
    llm = _llm(first, second, _text_turn("You have 3 open tasks.", tokens=6, label="3"))

    result = await _service(llm, provider=task_provider).generate_response(
        assistant, [], "What is open?", owner_user_id=OWNER, **BLOCKS
    )

    assert result.text == "You have 3 open tasks."
    assert result.tokens == 15
    assert result.debug.total_tokens == 15
    assert [step.iteration for step in result.debug.request_flow] == [1, 2, 3]
    assert [step.tokens for step in result.debug.request_flow] == [4, 5, 6]

    rejected = result.debug.request_flow[0].function_calls[0]
    assert rejected.name == "query_tasks"
    assert rejected.call_id == "call-1"
    assert rejected.arguments == {"unexpected": True}
    assert rejected.result["status"] == "filter_error"
    assert rejected.result["data"]["error_code"] == "invalid_arguments"
    accepted = result.debug.request_flow[1].function_calls[0]
    assert accepted.result == {"status": "ok", "data": {"total_count": 3}}
    assert result.debug.request_flow[2].assistant_text == "You have 3 open tasks."

    # The second request shows the model its own call and the error it caused.
    second_request = _sent_messages(llm, 1)
    assert second_request[-2] is first.response_event
    assert _is_tool_results(second_request[-1])
    fed_back = second_request[-1]["tool_results"]
    assert [item["name"] for item in fed_back] == ["query_tasks"]
    assert json.loads(fed_back[0]["content"]) == rejected.result

    task_provider.query_tasks.assert_awaited_once()
    assert task_provider.query_tasks.await_args.kwargs["owner_user_id"] == OWNER


async def test_tool_results_are_sent_as_json_text_in_call_order(assistant):
    calls = [
        LLMToolCall(name="list_projects", arguments={}, call_id="a"),
        LLMToolCall(name="query_tasks", arguments={"project": "Café"}, call_id="b"),
    ]
    dispatcher = AsyncMock(spec=ToolDispatcher)
    dispatcher.dispatch.return_value = [
        LLMToolResult(name="list_projects", content={"status": "no_data", "data": {}}),
        LLMToolResult(name="query_tasks", content={"status": "ok", "data": {"project": "Café"}}),
    ]
    llm = _llm(_tool_turn(*calls), _text_turn("Done."))

    result = await _service(llm, dispatcher=dispatcher).generate_response(
        assistant, [], "Overview, please", owner_user_id=OWNER, **BLOCKS
    )

    event = _sent_messages(llm, 1)[-1]
    assert event == {
        "tool_results": [
            {"name": "list_projects", "content": '{"status": "no_data", "data": {}}'},
            {
                "name": "query_tasks",
                "content": '{"status": "ok", "data": {"project": "Café"}}',
            },
        ]
    }
    assert [call.call_id for call in result.debug.request_flow[0].function_calls] == ["a", "b"]


async def test_owner_id_reaches_the_dispatcher_as_a_keyword(assistant):
    call = LLMToolCall(name="list_projects", arguments={}, call_id="a")
    dispatcher = AsyncMock(spec=ToolDispatcher)
    dispatcher.dispatch.return_value = [
        LLMToolResult(name="list_projects", content={"status": "no_data", "data": {}})
    ]
    llm = _llm(_tool_turn(call), _text_turn("Done."))

    await _service(llm, dispatcher=dispatcher).generate_response(
        assistant, [], "Which projects exist?", owner_user_id=OWNER, **BLOCKS
    )

    dispatcher.dispatch.assert_awaited_once_with([call], owner_user_id=OWNER)


async def test_window_keeps_the_system_message_and_the_most_recent_turns(assistant):
    llm = _llm(_text_turn("Done."))
    history = [
        _stored(f"message-{index}", MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT)
        for index in range(30)
    ]

    await _service(llm, history_limit=20).generate_response(
        assistant, history, "new-question", owner_user_id=OWNER, **BLOCKS
    )

    sent = _sent_messages(llm)
    # The system message does not count against the limit.
    assert len(sent) == 21
    assert sent[0]["role"] == "system"
    assert [item["text"] for item in sent[1:]] == [
        *[f"message-{index}" for index in range(11, 30)],
        "new-question",
    ]


async def test_window_never_separates_tool_results_from_the_call_that_requested_them(assistant):
    """With a tight limit the older turns go first; the running turn stays whole."""
    call = LLMToolCall(name="list_projects", arguments={}, call_id="a")
    turns = [_tool_turn(call, label=str(number)) for number in range(1, 4)]
    llm = _llm(*turns, _text_turn("Done.", label="4"))
    history = [
        _stored(f"message-{index}", MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT)
        for index in range(4)
    ]

    await _service(llm, provider=None, history_limit=3).generate_response(
        assistant, history, "the question", owner_user_id=OWNER, **BLOCKS
    )

    requests = [_sent_messages(llm, index) for index in range(4)]
    question = {"role": "user", "text": "the question"}

    # First request: the question plus the two older turns that still fit.
    assert requests[0][1:] == [
        {"role": "user", "text": "message-2"},
        {"role": "assistant", "text": "message-3"},
        question,
    ]
    # From then on the running turn alone fills (and then exceeds) the limit.
    assert len(requests[1]) == 1 + 3
    assert requests[1][1] == question
    assert requests[1][2] is turns[0].response_event
    assert _is_tool_results(requests[1][3])
    assert len(requests[2]) == 1 + 5
    assert len(requests[3]) == 1 + 7
    assert requests[3][2::2] == [turn.response_event for turn in turns]

    for request in requests:
        assert request[0]["role"] == "system"
        assert question in request
        assert not _is_tool_results(request[1])
        for position, item in enumerate(request):
            if _is_tool_results(item):
                assert isinstance(request[position - 1], _ResponseEvent)


async def test_each_request_is_a_separate_list(assistant):
    call = LLMToolCall(name="list_projects", arguments={}, call_id="a")
    llm = _llm(_tool_turn(call), _text_turn("Done."))

    await _service(llm).generate_response(
        assistant, [], "Which projects exist?", owner_user_id=OWNER, **BLOCKS
    )

    first, second = _sent_messages(llm, 0), _sent_messages(llm, 1)
    assert first is not second
    # What was sent first did not grow while the loop went on.
    assert len(first) == 2
    assert len(second) == 4


async def test_fallback_text_after_the_iteration_cap(assistant):
    call = LLMToolCall(name="unknown_tool", arguments={}, call_id="loop-call")
    llm = _llm(*[_tool_turn(call, tokens=2, label=str(number)) for number in range(5)])

    result = await _service(llm, max_iterations=5).generate_response(
        assistant, [], "Loop until the cap", owner_user_id=OWNER, **BLOCKS
    )

    assert result.text == AGENT_LOOP_FALLBACK_TEXT
    assert "could not finish" in result.text
    assert llm.run_agent_turn.await_count == 5
    # The trace survives: it shows what the model kept trying.
    assert result.tokens == 10
    assert [step.iteration for step in result.debug.request_flow] == [1, 2, 3, 4, 5]
    assert all(
        step.function_calls[0].result == {"error": "Unknown tool: unknown_tool"}
        for step in result.debug.request_flow
    )


async def test_turn_without_text_and_without_tool_calls_asks_the_model_again(assistant):
    llm = _llm(_text_turn("   ", label="blank"), _text_turn("Here you go.", label="answer"))

    result = await _service(llm).generate_response(
        assistant, [], "What is open?", owner_user_id=OWNER, **BLOCKS
    )

    assert result.text == "Here you go."
    assert llm.run_agent_turn.await_count == 2
    assert [step.assistant_text for step in result.debug.request_flow] == ["   ", "Here you go."]


async def test_result_count_mismatch_is_a_programming_error(assistant):
    calls = [
        LLMToolCall(name="list_projects", arguments={}, call_id="a"),
        LLMToolCall(name="list_projects", arguments={}, call_id="b"),
    ]
    dispatcher = AsyncMock(spec=ToolDispatcher)
    dispatcher.dispatch.return_value = [
        LLMToolResult(name="list_projects", content={"status": "no_data", "data": {}})
    ]
    llm = _llm(_tool_turn(*calls))

    with pytest.raises(ValueError, match=r"1 result\(s\) for 2 tool call\(s\)"):
        await _service(llm, dispatcher=dispatcher).generate_response(
            assistant, [], "Which projects exist?", owner_user_id=OWNER, **BLOCKS
        )


async def test_tool_calls_without_a_response_event_run_nothing(assistant):
    call = LLMToolCall(name="delete_task", arguments={"task": "Buy paint"}, call_id="a")
    dispatcher = AsyncMock(spec=ToolDispatcher)
    llm = _llm(LLMAgentTurnResult(text="", tool_calls=[call], tokens=1, response_event=None))

    with pytest.raises(ValueError, match="without a response event"):
        await _service(llm, dispatcher=dispatcher).generate_response(
            assistant, [], "Delete the paint task", owner_user_id=OWNER, **BLOCKS
        )

    dispatcher.dispatch.assert_not_awaited()


async def test_llm_service_error_is_left_to_the_caller(assistant):
    llm = AsyncMock(spec=LLMClient)
    llm.run_agent_turn.side_effect = LLMServiceUnavailableError("provider is down")

    with pytest.raises(LLMServiceUnavailableError):
        await _service(llm).generate_response(
            assistant, [], "What is open?", owner_user_id=OWNER, **BLOCKS
        )


async def test_costs_reported_by_the_client_are_summed(assistant):
    call = LLMToolCall(name="list_projects", arguments={}, call_id="a")
    llm = _llm(
        LLMAgentTurnResult(
            text="", tool_calls=[call], tokens=3, cost=0.25, response_event=_ResponseEvent("1")
        ),
        LLMAgentTurnResult(text="Done.", tokens=4, cost=0.5, response_event=_ResponseEvent("2")),
    )

    result = await _service(llm).generate_response(
        assistant, [], "Which projects exist?", owner_user_id=OWNER, **BLOCKS
    )

    assert result.tokens == 7
    assert result.cost == 0.75


@pytest.mark.parametrize(
    ("max_iterations", "history_limit", "message"),
    [(0, 20, "max_iterations"), (5, 0, "history_limit")],
)
def test_limits_below_one_are_rejected(max_iterations, history_limit, message):
    with pytest.raises(ValueError, match=message):
        _service(
            AsyncMock(spec=LLMClient),
            max_iterations=max_iterations,
            history_limit=history_limit,
        )
