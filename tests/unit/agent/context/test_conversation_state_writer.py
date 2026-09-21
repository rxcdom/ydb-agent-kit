"""Tests for the conversation-state writer: what a turn's trace leaves behind."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from src.agent.application.context.conversation_state_writer import ConversationStateWriter
from src.agent.domain.entities.chat_agent_context import ChatAgentContext
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.ports.llm_tooling import (
    AgentFunctionCallTrace,
    AgentIterationTrace,
    AgentRequestDebugTrace,
)
from src.shared.domain.exceptions import PersistenceError
from src.shared.domain.value_objects.user_id import UserId


def _trace(*iterations: list[AgentFunctionCallTrace]) -> AgentRequestDebugTrace:
    return AgentRequestDebugTrace(
        model_name="test-model",
        total_tokens=0,
        request_flow=[
            AgentIterationTrace(
                iteration=index, tokens=0, assistant_text="", function_calls=list(calls)
            )
            for index, calls in enumerate(iterations, start=1)
        ],
    )


def _ok_query(
    window_from: str,
    window_to: str,
    *,
    date_field: str = "created",
    project: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> AgentFunctionCallTrace:
    data: Dict[str, Any] = {
        "window_used": {"from": window_from, "to": window_to},
        "coverage": {"first": "2026-01-05", "last": "2026-09-17"},
        "date_field": date_field,
        "total_count": 4,
        "excluded_without_date": 0,
    }
    if project is not None:
        data["project_resolved"] = {"name": project}
    return AgentFunctionCallTrace(
        name="query_tasks", arguments=arguments or {}, result={"status": "ok", "data": data}
    )


def _failed_query(status: str) -> AgentFunctionCallTrace:
    return AgentFunctionCallTrace(
        name="query_tasks",
        arguments={"project": "Home"},
        result={"status": status, "data": {"coverage": {"first": "2026-01-05"}}},
    )


@pytest.fixture
def writer(mock_repository_manager) -> ConversationStateWriter:
    return ConversationStateWriter(mock_repository_manager)


async def test_last_ok_query_of_the_turn_wins(writer, mock_repository_manager):
    chat_id, user_id = ChatId.generate(), UserId.generate()
    trace = _trace(
        [_ok_query("2026-04-01", "2026-04-07")],
        [_ok_query("2026-09-01", "2026-09-17", date_field="due", project="Home renovation")],
    )

    await writer.record_turn(chat_id=chat_id, user_id=user_id, trace=trace)

    mock_repository_manager.chat_agent_context.find_by_chat_id.assert_awaited_once_with(chat_id)
    mock_repository_manager.chat_agent_context.save.assert_awaited_once()
    saved: ChatAgentContext = mock_repository_manager.chat_agent_context.save.await_args.args[0]
    assert saved.chat_id == chat_id
    assert saved.user_id == user_id
    assert saved.last_tool == "query_tasks"
    assert saved.last_window_from == "2026-09-01"
    assert saved.last_window_to == "2026-09-17"
    assert saved.last_date_field == "due"
    assert saved.last_project == "Home renovation"


async def test_window_comes_from_the_result_not_from_the_arguments(
    writer, mock_repository_manager
):
    """The tool resolves open bounds; the state keeps what was actually used."""
    trace = _trace(
        [_ok_query("2026-01-05", "2026-09-16", arguments={"date_to": "2026-09-16"})]
    )

    await writer.record_turn(chat_id=ChatId.generate(), user_id=UserId.generate(), trace=trace)

    saved = mock_repository_manager.chat_agent_context.save.await_args.args[0]
    assert (saved.last_window_from, saved.last_window_to) == ("2026-01-05", "2026-09-16")


@pytest.mark.parametrize(
    "status",
    ["no_data", "coverage_gap", "empty_filter", "no_records", "ambiguous_source", "filter_error"],
)
async def test_failed_outcome_preserves_the_prior_state(
    writer, mock_repository_manager, status
):
    chat_id, user_id = ChatId.generate(), UserId.generate()
    prior = ChatAgentContext.empty_for(chat_id, user_id)
    prior.record_windowed_tool(
        tool="query_tasks",
        window_from="2026-08-01",
        window_to="2026-08-31",
        date_field="completed",
        project="Kitchen",
    )
    mock_repository_manager.chat_agent_context.find_by_chat_id.return_value = prior

    await writer.record_turn(
        chat_id=chat_id, user_id=user_id, trace=_trace([_failed_query(status)])
    )

    mock_repository_manager.chat_agent_context.save.assert_not_awaited()
    assert prior.last_window_from == "2026-08-01"
    assert prior.last_project == "Kitchen"


async def test_failed_probe_after_a_good_query_keeps_the_good_one(
    writer, mock_repository_manager
):
    trace = _trace(
        [_ok_query("2026-09-01", "2026-09-17", project="Kitchen")],
        [_failed_query("coverage_gap")],
    )

    await writer.record_turn(chat_id=ChatId.generate(), user_id=UserId.generate(), trace=trace)

    saved = mock_repository_manager.chat_agent_context.save.await_args.args[0]
    assert saved.last_window_from == "2026-09-01"
    assert saved.last_project == "Kitchen"


async def test_turn_without_a_windowed_call_is_a_no_op(writer, mock_repository_manager):
    trace = _trace(
        [
            AgentFunctionCallTrace(
                name="list_projects", arguments={}, result={"status": "ok", "data": {}}
            ),
            AgentFunctionCallTrace(
                name="recall", arguments={}, result={"status": "ok", "count": 0, "memories": []}
            ),
            AgentFunctionCallTrace(
                name="update_task",
                arguments={"task": "Buy paint", "set_status": "done"},
                result={"status": "ok", "data": {"changed": []}},
            ),
        ]
    )

    await writer.record_turn(chat_id=ChatId.generate(), user_id=UserId.generate(), trace=trace)

    mock_repository_manager.chat_agent_context.find_by_chat_id.assert_not_awaited()
    mock_repository_manager.chat_agent_context.save.assert_not_awaited()


async def test_turn_without_tool_calls_is_a_no_op(writer, mock_repository_manager):
    await writer.record_turn(
        chat_id=ChatId.generate(), user_id=UserId.generate(), trace=_trace([])
    )

    mock_repository_manager.chat_agent_context.save.assert_not_awaited()


async def test_existing_state_is_updated_in_place(writer, mock_repository_manager):
    chat_id, user_id = ChatId.generate(), UserId.generate()
    existing = ChatAgentContext.empty_for(chat_id, user_id)
    existing.record_windowed_tool(
        tool="query_tasks",
        window_from="2026-01-01",
        window_to="2026-01-07",
        date_field="created",
        project="Kitchen",
    )
    created_at = existing.created_at
    mock_repository_manager.chat_agent_context.find_by_chat_id.return_value = existing

    await writer.record_turn(
        chat_id=chat_id,
        user_id=user_id,
        trace=_trace([_ok_query("2026-09-01", "2026-09-17", date_field="completed")]),
    )

    saved = mock_repository_manager.chat_agent_context.save.await_args.args[0]
    assert saved is existing
    assert saved.created_at == created_at
    assert saved.last_window_from == "2026-09-01"
    assert saved.last_date_field == "completed"
    # The new query had no project filter, so none is active any more.
    assert saved.last_project is None


@pytest.mark.parametrize(
    "result",
    [
        {"status": "ok"},
        {"status": "ok", "data": None},
        {"status": "ok", "data": {"window_used": None}},
        {"status": "ok", "data": {"window_used": "2026-09-01..2026-09-17"}},
        {"status": "ok", "data": {"window_used": {"from": None, "to": ""}}},
        {"status": "ok", "data": {"window_used": {"from": 20260901, "to": 20260917}}},
    ],
)
async def test_ok_result_without_a_usable_window_is_skipped(
    writer, mock_repository_manager, result
):
    trace = _trace(
        [AgentFunctionCallTrace(name="query_tasks", arguments={}, result=result)]
    )

    await writer.record_turn(chat_id=ChatId.generate(), user_id=UserId.generate(), trace=trace)

    mock_repository_manager.chat_agent_context.save.assert_not_awaited()


async def test_malformed_optional_values_count_as_absent(writer, mock_repository_manager):
    result = {
        "status": "ok",
        "data": {
            "window_used": {"from": "2026-09-01", "to": "2026-09-17"},
            "date_field": ["created"],
            "project_resolved": "Kitchen",
        },
    }
    trace = _trace([AgentFunctionCallTrace(name="query_tasks", arguments={}, result=result)])

    await writer.record_turn(chat_id=ChatId.generate(), user_id=UserId.generate(), trace=trace)

    saved = mock_repository_manager.chat_agent_context.save.await_args.args[0]
    assert saved.last_window_from == "2026-09-01"
    assert saved.last_date_field is None
    assert saved.last_project is None


async def test_value_the_entity_refuses_leaves_the_prior_state_in_place(
    writer, mock_repository_manager
):
    chat_id, user_id = ChatId.generate(), UserId.generate()
    prior = ChatAgentContext.empty_for(chat_id, user_id)
    prior.record_windowed_tool(
        tool="query_tasks",
        window_from="2026-08-01",
        window_to="2026-08-31",
        date_field="created",
        project="Kitchen",
    )
    mock_repository_manager.chat_agent_context.find_by_chat_id.return_value = prior
    trace = _trace([_ok_query("2026-09-01", "2026-09-17", project="P" * 200)])

    await writer.record_turn(chat_id=chat_id, user_id=user_id, trace=trace)

    mock_repository_manager.chat_agent_context.save.assert_not_awaited()
    assert prior.last_window_from == "2026-08-01"
    assert prior.last_project == "Kitchen"


async def test_storage_failure_is_raised_to_the_caller(writer, mock_repository_manager):
    """The writer does not decide that it is optional; its caller does."""
    mock_repository_manager.chat_agent_context.save.side_effect = PersistenceError("down")

    with pytest.raises(PersistenceError):
        await writer.record_turn(
            chat_id=ChatId.generate(),
            user_id=UserId.generate(),
            trace=_trace([_ok_query("2026-09-01", "2026-09-17")]),
        )
