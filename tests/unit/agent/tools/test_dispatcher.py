"""Tests for the tool dispatcher: validation, fallbacks, ordering and logging."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Literal, Optional
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ConfigDict, Field

from src.agent.application.tools import build_tool_registry
from src.agent.application.tools.dispatcher import LOG_VALUE_CAP, ToolDispatcher
from src.agent.application.tools.registry import ToolRegistry
from src.agent.ports.llm_tooling import LLMToolCall, LLMToolDefinition, LLMToolResult
from src.agent.ports.task_data_provider import TaskDataProvider

DISPATCHER_LOGGER = "src.agent.application.tools.dispatcher"
OWNER = "owner-1"


class _NoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    colour: Literal["red", "green"] = "green"
    day: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _definition(name: str, model: type[BaseModel]) -> LLMToolDefinition:
    return LLMToolDefinition(
        name=name, description=f"Test tool {name}.", parameters=model.model_json_schema()
    )


def _dispatcher_with(name: str, model: type[BaseModel], executor) -> ToolDispatcher:
    registry = ToolRegistry()
    registry.register(_definition(name, model), model, executor)
    return ToolDispatcher(registry)


def _note_executor(calls: Optional[List[dict]] = None):
    async def execute(args: _NoteArgs, *, owner_user_id: str) -> LLMToolResult:
        if calls is not None:
            calls.append({"args": args, "owner_user_id": owner_user_id})
        return LLMToolResult(
            name="note", content={"status": "ok", "data": {"text": args.text}}
        )

    return execute


def _task_provider() -> AsyncMock:
    provider = AsyncMock(spec=TaskDataProvider)
    provider.query_tasks.return_value = {"status": "ok", "data": {"total_count": 3}}
    return provider


async def test_unknown_tool_returns_error_result():
    dispatcher = ToolDispatcher(ToolRegistry())

    results = await dispatcher.dispatch(
        [LLMToolCall(name="does_not_exist", arguments={}, call_id="c1")],
        owner_user_id=OWNER,
    )

    assert results == [
        LLMToolResult(name="does_not_exist", content={"error": "Unknown tool: does_not_exist"})
    ]


async def test_empty_call_list_returns_no_results():
    assert await ToolDispatcher(ToolRegistry()).dispatch([], owner_user_id=OWNER) == []


async def test_valid_call_runs_the_executor_with_validated_arguments():
    calls: List[dict] = []
    dispatcher = _dispatcher_with("note", _NoteArgs, _note_executor(calls))

    results = await dispatcher.dispatch(
        [LLMToolCall(name="note", arguments={"text": "Buy paint"}, call_id="c1")],
        owner_user_id=OWNER,
    )

    assert results[0].content == {"status": "ok", "data": {"text": "Buy paint"}}
    assert calls == [
        {"args": _NoteArgs(text="Buy paint", colour="green", day=None), "owner_user_id": OWNER}
    ]


async def test_owner_id_reaches_the_executor_as_a_keyword():
    executor = AsyncMock(return_value=LLMToolResult(name="note", content={"status": "ok"}))
    dispatcher = _dispatcher_with("note", _NoteArgs, executor)

    await dispatcher.dispatch(
        [LLMToolCall(name="note", arguments={"text": "Buy paint"}, call_id="c1")],
        owner_user_id=OWNER,
    )

    assert executor.await_args.args == (_NoteArgs(text="Buy paint"),)
    assert executor.await_args.kwargs == {"owner_user_id": OWNER}


async def test_null_optionals_are_folded_onto_the_defaults():
    """A model that sends null for every argument it does not use still gets an answer."""
    provider = _task_provider()
    dispatcher = ToolDispatcher(build_tool_registry(provider, None))

    results = await dispatcher.dispatch(
        [
            LLMToolCall(
                name="query_tasks",
                arguments={
                    "date_from": None,
                    "date_to": None,
                    "date_field": None,
                    "project": None,
                    "statuses": ["open"],
                    "priority": None,
                    "text": None,
                    "group_by": None,
                    "view": None,
                },
                call_id="c1",
            )
        ],
        owner_user_id=OWNER,
    )

    assert results[0].content["status"] == "ok"
    kwargs = provider.query_tasks.await_args.kwargs
    assert kwargs["date_field"] == "created"
    assert kwargs["group_by"] == "none"
    assert kwargs["view"] == "summary"
    assert kwargs["statuses"] == ["open"]
    assert kwargs["date_from"] is None


async def test_null_for_a_required_argument_is_reported_as_required():
    calls: List[dict] = []
    dispatcher = _dispatcher_with("note", _NoteArgs, _note_executor(calls))

    results = await dispatcher.dispatch(
        [LLMToolCall(name="note", arguments={"text": None}, call_id="c1")],
        owner_user_id=OWNER,
    )

    assert results[0].content["status"] == "filter_error"
    assert results[0].content["data"]["message"] == "text is required"
    assert calls == []


async def test_schema_error_uses_the_filter_error_envelope_and_names_each_field():
    calls: List[dict] = []
    dispatcher = _dispatcher_with("note", _NoteArgs, _note_executor(calls))

    results = await dispatcher.dispatch(
        [
            LLMToolCall(
                name="note",
                arguments={"colour": "blue", "day": "tomorrow", "size": 3},
                call_id="c1",
            )
        ],
        owner_user_id=OWNER,
    )

    content = results[0].content
    assert set(content) == {"status", "data"}
    assert content["status"] == "filter_error"
    assert set(content["data"]) == {"error_code", "message"}
    assert content["data"]["error_code"] == "invalid_arguments"

    problems = content["data"]["message"].split("; ")
    assert "text is required" in problems
    assert "colour must be one of: 'red' or 'green'" in problems
    assert any(problem.startswith("day: String should match pattern") for problem in problems)
    assert (
        "size is not an argument of this tool (allowed arguments: text, colour, day)" in problems
    )
    assert len(problems) == 4
    # Nothing of the validation library leaks into the conversation.
    assert "pydantic" not in content["data"]["message"]
    assert calls == []


async def test_unexpected_argument_of_a_tool_without_arguments():
    executor = AsyncMock(return_value=LLMToolResult(name="ping", content={"status": "ok"}))
    dispatcher = _dispatcher_with("ping", _NoArgs, executor)

    results = await dispatcher.dispatch(
        [LLMToolCall(name="ping", arguments={"verbose": True}, call_id="c1")],
        owner_user_id=OWNER,
    )

    assert results[0].content["data"]["message"] == (
        "verbose is not an argument of this tool (it takes no arguments)"
    )
    executor.assert_not_awaited()


async def test_list_item_outside_the_vocabulary_is_located_by_index():
    provider = _task_provider()
    dispatcher = ToolDispatcher(build_tool_registry(provider, None))

    results = await dispatcher.dispatch(
        [LLMToolCall(name="query_tasks", arguments={"statuses": ["open", "late"]}, call_id="c1")],
        owner_user_id=OWNER,
    )

    assert results[0].content["data"]["message"] == (
        "statuses.1 must be one of: 'open', 'done' or 'cancelled'"
    )
    provider.query_tasks.assert_not_awaited()


async def test_arguments_that_are_not_an_object_are_a_schema_error():
    dispatcher = _dispatcher_with("note", _NoteArgs, _note_executor())

    results = await dispatcher.dispatch(
        [LLMToolCall(name="note", arguments="Buy paint", call_id="c1")],  # type: ignore[arg-type]
        owner_user_id=OWNER,
    )

    assert results[0].content == {
        "status": "filter_error",
        "data": {
            "error_code": "invalid_arguments",
            "message": "arguments must be a JSON object",
        },
    }


async def test_executor_exception_becomes_an_error_result_and_is_logged_with_traceback(caplog):
    async def failing(args: _NoteArgs, *, owner_user_id: str) -> LLMToolResult:
        raise RuntimeError("connection string with internal details")

    dispatcher = _dispatcher_with("note", _NoteArgs, failing)

    with caplog.at_level(logging.ERROR, logger=DISPATCHER_LOGGER):
        results = await dispatcher.dispatch(
            [LLMToolCall(name="note", arguments={"text": "Buy paint"}, call_id="c1")],
            owner_user_id=OWNER,
        )

    assert results == [
        LLMToolResult(name="note", content={"error": "tool_execution_failed: RuntimeError"})
    ]
    failure_records = [
        record for record in caplog.records if "Tool execution failed: note" in record.getMessage()
    ]
    assert len(failure_records) == 1
    assert failure_records[0].exc_info is not None
    assert failure_records[0].exc_info[0] is RuntimeError


@pytest.mark.parametrize(
    "returned",
    [
        None,
        {"status": "ok"},
        LLMToolResult(name="note", content=["not", "an", "object"]),  # type: ignore[arg-type]
        LLMToolResult(name="note", content={"at": datetime(2026, 9, 1, tzinfo=timezone.utc)}),
    ],
    ids=["none", "bare-dict", "list-content", "non-json-value"],
)
async def test_executor_result_that_is_not_a_json_object_is_a_tool_failure(returned):
    executor = AsyncMock(return_value=returned)
    dispatcher = _dispatcher_with("note", _NoteArgs, executor)

    results = await dispatcher.dispatch(
        [LLMToolCall(name="note", arguments={"text": "Buy paint"}, call_id="c1")],
        owner_user_id=OWNER,
    )

    assert results == [
        LLMToolResult(name="note", content={"error": "tool_execution_failed: TypeError"})
    ]


async def test_every_call_gets_one_result_in_order_whatever_happens():
    async def failing(args: _NoArgs, *, owner_user_id: str) -> LLMToolResult:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(_definition("note", _NoteArgs), _NoteArgs, _note_executor())
    registry.register(_definition("broken", _NoArgs), _NoArgs, failing)
    dispatcher = ToolDispatcher(registry)

    results = await dispatcher.dispatch(
        [
            LLMToolCall(name="note", arguments={"text": "first"}, call_id="c1"),
            LLMToolCall(name="missing", arguments={}, call_id="c2"),
            LLMToolCall(name="note", arguments={}, call_id="c3"),
            LLMToolCall(name="broken", arguments={}, call_id="c4"),
            LLMToolCall(name="note", arguments={"text": "last"}, call_id="c5"),
        ],
        owner_user_id=OWNER,
    )

    assert [result.name for result in results] == ["note", "missing", "note", "broken", "note"]
    assert results[0].content["data"]["text"] == "first"
    assert results[1].content == {"error": "Unknown tool: missing"}
    assert results[2].content["status"] == "filter_error"
    assert results[3].content == {"error": "tool_execution_failed: RuntimeError"}
    # A failure earlier in the turn does not stop the calls after it.
    assert results[4].content["data"]["text"] == "last"


async def test_debug_log_shows_tool_name_payload_and_result(caplog):
    provider = _task_provider()
    dispatcher = ToolDispatcher(build_tool_registry(provider, None))

    with caplog.at_level(logging.DEBUG, logger=DISPATCHER_LOGGER):
        await dispatcher.dispatch(
            [
                LLMToolCall(
                    name="query_tasks",
                    arguments={"project": "Kitchen", "date_from": "2026-09-01"},
                    call_id="c1",
                )
            ],
            owner_user_id=OWNER,
        )

    log = caplog.text
    assert "Tool turn: 1 call(s): ['query_tasks']" in log
    assert "Tool call query_tasks (id=c1) payload:" in log
    assert "Kitchen" in log
    assert "Tool result query_tasks (id=c1) content:" in log
    assert "total_count" in log


async def test_info_log_names_arguments_and_outcome_without_their_values(caplog):
    provider = _task_provider()
    dispatcher = ToolDispatcher(build_tool_registry(provider, None))

    with caplog.at_level(logging.INFO, logger=DISPATCHER_LOGGER):
        await dispatcher.dispatch(
            [
                LLMToolCall(
                    name="query_tasks",
                    arguments={"text": "dentist appointment", "view": "list"},
                    call_id="c1",
                )
            ],
            owner_user_id=OWNER,
        )

    log = caplog.text
    assert "Tool call query_tasks (id=c1): arguments=['text', 'view']" in log
    assert "Tool result query_tasks (id=c1): status=ok" in log
    assert "dentist appointment" not in log
    assert "total_count" not in log


async def test_long_payload_is_capped_in_the_log(caplog):
    dispatcher = _dispatcher_with("note", _NoteArgs, _note_executor())
    long_text = "x" * (LOG_VALUE_CAP * 3)

    with caplog.at_level(logging.DEBUG, logger=DISPATCHER_LOGGER):
        await dispatcher.dispatch(
            [LLMToolCall(name="note", arguments={"text": long_text}, call_id="c1")],
            owner_user_id=OWNER,
        )

    payload_records = [
        record for record in caplog.records if "payload:" in record.getMessage()
    ]
    assert len(payload_records) == 1
    message = payload_records[0].getMessage()
    assert "more characters]" in message
    assert long_text not in message
    assert len(message) < LOG_VALUE_CAP + 200
