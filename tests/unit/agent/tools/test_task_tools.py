"""Tests for the task tool executors and for the assembled tool registry."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.agent.application.tools import TOOL_DEFINITIONS, build_tool_registry
from src.agent.application.tools.dispatcher import ToolDispatcher
from src.agent.application.tools.schemas import (
    ALL_ARGS_MODELS,
    CreateTaskArgs,
    DeleteTaskArgs,
    ListProjectsArgs,
    QueryTasksArgs,
    UpdateTaskArgs,
)
from src.agent.application.tools.task_tools import TASK_TOOL_NAMES, build_task_tool_executors
from src.agent.ports.llm_tooling import LLMToolCall
from src.agent.ports.task_data_provider import TaskDataProvider

OWNER = "0b5c1f0e-6f0a-4a57-9d55-3f1f4a1c2b10"

EXPECTED_TOOL_NAMES = [
    "list_projects",
    "query_tasks",
    "create_task",
    "update_task",
    "delete_task",
    "remember",
    "recall",
    "forget",
]

# One envelope per word of the outcome vocabulary the provider may answer with.
ENVELOPES = [
    {
        "status": "ok",
        "data": {
            "window_used": {"from": "2026-09-01", "to": "2026-09-17"},
            "coverage": {"first": "2026-01-05", "last": "2026-09-17"},
            "date_field": "created",
            "total_count": 2,
            "excluded_without_date": 0,
            "tasks": [{"title": "Buy paint"}, {"title": "Order tiles"}],
        },
    },
    {"status": "no_data", "data": {"projects_count": 0}},
    {
        "status": "coverage_gap",
        "data": {
            "coverage": {"first": "2026-01-05", "last": "2026-09-17"},
            "requested_window": {"from": "2025-01-01", "to": "2025-01-31"},
            "date_field": "created",
        },
    },
    {
        "status": "empty_filter",
        "data": {
            "window_used": {"from": "2026-09-01", "to": "2026-09-17"},
            "count_without_filters": 7,
            "available_projects": ["Home", "Home renovation"],
            "available_statuses": ["open", "done"],
            "note": "No task matched the filters.",
        },
    },
    {
        "status": "no_records",
        "data": {
            "window_used": {"from": "2026-07-01", "to": "2026-07-14"},
            "count_without_window": 58,
        },
    },
    {
        "status": "ambiguous_source",
        "data": {
            "reference": "Home",
            "matched": [{"name": "Home"}, {"name": "Home renovation"}],
            "note": "Ask the user which one is meant.",
        },
    },
    {"status": "not_found", "data": {"reference": "Garage", "note": "Nothing matches."}},
    {
        "status": "filter_error",
        "data": {"error_code": "contradictory_filters", "message": "completed excludes open"},
    },
]


def _provider(envelope: dict | None = None) -> AsyncMock:
    provider = AsyncMock(spec=TaskDataProvider)
    answer = envelope if envelope is not None else {"status": "ok", "data": {}}
    for method in ("list_projects", "query_tasks", "create_task", "update_task", "delete_task"):
        getattr(provider, method).return_value = answer
    return provider


@pytest.mark.parametrize("envelope", ENVELOPES, ids=[item["status"] for item in ENVELOPES])
async def test_provider_envelope_is_the_tool_result_unchanged(envelope: dict):
    provider = _provider(envelope)
    executors = build_task_tool_executors(provider)

    result = await executors["query_tasks"](QueryTasksArgs(), owner_user_id=OWNER)

    assert result.name == "query_tasks"
    assert result.content is envelope


async def test_every_task_tool_names_its_own_result():
    executors = build_task_tool_executors(_provider())
    arguments = {
        "list_projects": ListProjectsArgs(),
        "query_tasks": QueryTasksArgs(),
        "create_task": CreateTaskArgs(title="Buy paint"),
        "update_task": UpdateTaskArgs(task="Buy paint"),
        "delete_task": DeleteTaskArgs(task="Buy paint"),
    }

    assert set(executors) == set(TASK_TOOL_NAMES) == set(arguments)
    for name, args in arguments.items():
        result = await executors[name](args, owner_user_id=OWNER)
        assert result.name == name
        assert result.content == {"status": "ok", "data": {}}


async def test_list_projects_passes_only_the_owner():
    provider = _provider()

    await build_task_tool_executors(provider)["list_projects"](
        ListProjectsArgs(), owner_user_id=OWNER
    )

    assert provider.list_projects.await_args.args == ()
    assert provider.list_projects.await_args.kwargs == {"owner_user_id": OWNER}


async def test_query_tasks_passes_every_argument_by_keyword():
    provider = _provider()
    args = QueryTasksArgs(
        date_from="2026-09-01",
        date_to="2026-09-17",
        date_field="due",
        project="Home renovation",
        statuses=["open"],
        priority=["high", "normal"],
        text="paint",
        group_by="week",
        view="list",
    )

    await build_task_tool_executors(provider)["query_tasks"](args, owner_user_id=OWNER)

    assert provider.query_tasks.await_args.args == ()
    assert provider.query_tasks.await_args.kwargs == {
        "owner_user_id": OWNER,
        "date_from": "2026-09-01",
        "date_to": "2026-09-17",
        "date_field": "due",
        "project": "Home renovation",
        "statuses": ["open"],
        "priority": ["high", "normal"],
        "text": "paint",
        "group_by": "week",
        "view": "list",
    }


async def test_query_tasks_defaults_reach_the_provider():
    provider = _provider()

    await build_task_tool_executors(provider)["query_tasks"](QueryTasksArgs(), owner_user_id=OWNER)

    assert provider.query_tasks.await_args.kwargs == {
        "owner_user_id": OWNER,
        "date_from": None,
        "date_to": None,
        "date_field": "created",
        "project": None,
        "statuses": None,
        "priority": None,
        "text": None,
        "group_by": "none",
        "view": "summary",
    }


async def test_create_task_passes_every_argument_by_keyword():
    provider = _provider()
    args = CreateTaskArgs(
        title="Order tiles",
        project="Home renovation",
        due_at="2026-10-01",
        priority="high",
        notes="Grey, 60 by 60",
    )

    await build_task_tool_executors(provider)["create_task"](args, owner_user_id=OWNER)

    assert provider.create_task.await_args.args == ()
    assert provider.create_task.await_args.kwargs == {
        "owner_user_id": OWNER,
        "title": "Order tiles",
        "project": "Home renovation",
        "due_at": "2026-10-01",
        "priority": "high",
        "notes": "Grey, 60 by 60",
    }


async def test_update_task_passes_every_argument_by_keyword():
    provider = _provider()
    args = UpdateTaskArgs(
        task="Order tiles",
        project_scope="Home renovation",
        set_status="done",
        set_title="Order floor tiles",
        set_due_at="none",
        set_priority="low",
        set_project="none",
        set_notes="Delivered",
    )

    await build_task_tool_executors(provider)["update_task"](args, owner_user_id=OWNER)

    assert provider.update_task.await_args.args == ()
    assert provider.update_task.await_args.kwargs == {
        "owner_user_id": OWNER,
        "task": "Order tiles",
        "project_scope": "Home renovation",
        "set_status": "done",
        "set_title": "Order floor tiles",
        "set_due_at": "none",
        "set_priority": "low",
        "set_project": "none",
        "set_notes": "Delivered",
    }


async def test_delete_task_passes_every_argument_by_keyword():
    provider = _provider()
    args = DeleteTaskArgs(task="Order tiles", project_scope="Home renovation")

    await build_task_tool_executors(provider)["delete_task"](args, owner_user_id=OWNER)

    assert provider.delete_task.await_args.args == ()
    assert provider.delete_task.await_args.kwargs == {
        "owner_user_id": OWNER,
        "task": "Order tiles",
        "project_scope": "Home renovation",
    }


@pytest.mark.parametrize("tool_name", TASK_TOOL_NAMES)
async def test_task_tools_degrade_when_the_provider_is_not_wired(tool_name: str):
    arguments = {
        "list_projects": {},
        "query_tasks": {},
        "create_task": {"title": "Buy paint"},
        "update_task": {"task": "Buy paint", "set_status": "done"},
        "delete_task": {"task": "Buy paint"},
    }
    dispatcher = ToolDispatcher(build_tool_registry(None, None))

    results = await dispatcher.dispatch(
        [LLMToolCall(name=tool_name, arguments=arguments[tool_name], call_id="c1")],
        owner_user_id=OWNER,
    )

    assert results[0].name == tool_name
    assert results[0].content == {"error": "task_data_unavailable"}


async def test_ambiguous_write_reaches_the_model_as_the_provider_reported_it():
    """The executor adds no judgement of its own: a refused write stays refused."""
    envelope = {
        "status": "ambiguous_source",
        "data": {
            "reference": "report",
            "matched": [
                {"title": "Quarterly report", "project": "Work", "status": "open"},
                {"title": "Expense report", "project": "Work", "status": "open"},
            ],
            "note": "Nothing was deleted.",
        },
    }
    provider = _provider(envelope)
    dispatcher = ToolDispatcher(build_tool_registry(provider, None))

    results = await dispatcher.dispatch(
        [LLMToolCall(name="delete_task", arguments={"task": "report"}, call_id="c1")],
        owner_user_id=OWNER,
    )

    assert results[0].content == envelope
    provider.delete_task.assert_awaited_once_with(
        owner_user_id=OWNER, task="report", project_scope=None
    )


def test_build_tool_registry_registers_all_eight_tools():
    registry = build_tool_registry(_provider(), AsyncMock())

    assert [definition.name for definition in registry.definitions()] == EXPECTED_TOOL_NAMES
    assert [definition.name for definition in TOOL_DEFINITIONS] == EXPECTED_TOOL_NAMES
    for spec in registry.specs():
        assert spec.args_model is ALL_ARGS_MODELS[spec.name]
        assert spec.definition.parameters == spec.args_model.model_json_schema()
        assert spec.definition.description.strip()
        assert callable(spec.executor)


def test_every_tool_stays_registered_without_collaborators():
    registry = build_tool_registry(None, None)

    assert [spec.name for spec in registry.specs()] == EXPECTED_TOOL_NAMES


def test_no_registered_args_model_carries_an_identity_field():
    """The owner comes from the principal and rows are addressed by human text."""
    forbidden = {"user_id", "owner_id", "owner_user_id", "chat_id", "memory_id"}
    registry = build_tool_registry(_provider(), AsyncMock())

    for spec in registry.specs():
        declared = set(spec.args_model.model_fields)
        offered = set(spec.definition.parameters.get("properties", {}))
        for names in (declared, offered):
            assert not names & forbidden, f"{spec.name} takes {names & forbidden}"
            # No argument is an identifier of any kind: not of a task or a
            # project either.
            identifiers = {name for name in names if name == "id" or name.endswith("_id")}
            assert not identifiers, f"{spec.name} takes {identifiers}"


def test_every_args_model_rejects_unknown_arguments():
    for name, model in ALL_ARGS_MODELS.items():
        assert model.model_config.get("extra") == "forbid", name
