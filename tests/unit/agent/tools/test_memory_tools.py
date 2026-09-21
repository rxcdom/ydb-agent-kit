"""Tests for the memory tool executors, reached through the dispatcher."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.agent.application.memory.user_memory_writer import (
    ForgetOutcome,
    RememberOutcome,
    UserMemoryWriter,
)
from src.agent.application.tools import build_tool_registry
from src.agent.application.tools.dispatcher import ToolDispatcher
from src.agent.application.tools.memory_tools import (
    MEMORY_TOOL_NAMES,
    build_memory_tool_executors,
)
from src.agent.domain.entities.user_memory import UserMemory
from src.agent.ports.llm_tooling import LLMToolCall
from src.shared.domain.value_objects.user_id import UserId


def _writer() -> AsyncMock:
    writer = AsyncMock(spec=UserMemoryWriter)
    writer.remember.return_value = RememberOutcome(status="created", memory_id="m-1")
    writer.recall.return_value = []
    writer.forget.return_value = ForgetOutcome(deleted_count=0)
    return writer


def _dispatcher(writer) -> ToolDispatcher:
    return ToolDispatcher(build_tool_registry(None, writer))


async def test_dispatcher_routes_memory_tools_to_the_writer_with_the_owner_scope():
    owner = UserId.generate()
    writer = _writer()
    writer.forget.return_value = ForgetOutcome(deleted_count=1)

    results = await _dispatcher(writer).dispatch(
        [
            LLMToolCall(
                name="remember",
                arguments={"content": "Reviews open tasks every Friday", "topic": "routine"},
                call_id="c1",
            ),
            LLMToolCall(name="recall", arguments={"query": "Friday"}, call_id="c2"),
            LLMToolCall(name="forget", arguments={"query": "Friday"}, call_id="c3"),
        ],
        owner_user_id=str(owner),
    )

    assert [result.name for result in results] == ["remember", "recall", "forget"]
    assert results[0].content == {"status": "ok", "action": "created", "evicted_oldest": False}
    assert results[1].content == {"status": "ok", "count": 0, "memories": []}
    assert results[2].content == {"status": "ok", "deleted_count": 1}

    # The vault is always the one of the authenticated owner.
    writer.remember.assert_awaited_once_with(
        owner, content="Reviews open tasks every Friday", topic="routine"
    )
    writer.recall.assert_awaited_once_with(owner, query="Friday")
    writer.forget.assert_awaited_once_with(owner, query="Friday")


async def test_remember_reports_an_update_and_an_eviction():
    writer = _writer()
    writer.remember.side_effect = [
        RememberOutcome(status="updated", memory_id="m-1"),
        RememberOutcome(status="created", memory_id="m-2", evicted=True),
    ]
    dispatcher = _dispatcher(writer)
    call = LLMToolCall(name="remember", arguments={"content": "Prefers morning deadlines"})

    updated = await dispatcher.dispatch([call], owner_user_id=str(UserId.generate()))
    created = await dispatcher.dispatch([call], owner_user_id=str(UserId.generate()))

    assert updated[0].content == {"status": "ok", "action": "updated", "evicted_oldest": False}
    assert created[0].content == {"status": "ok", "action": "created", "evicted_oldest": True}


async def test_remember_rejection_reaches_the_agent_with_its_reason():
    writer = _writer()
    writer.remember.return_value = RememberOutcome(
        status="rejected", reason="looks_like_secret"
    )

    results = await _dispatcher(writer).dispatch(
        [
            LLMToolCall(
                name="remember",
                arguments={"content": "My password is hunter2-example"},
                call_id="c1",
            )
        ],
        owner_user_id=str(UserId.generate()),
    )

    content = results[0].content
    assert content["status"] == "rejected"
    assert content["reason"] == "looks_like_secret"
    assert "Nothing was stored" in content["note"]
    assert set(content) == {"status", "reason", "note"}


async def test_recall_returns_content_topic_and_date_but_no_identifiers():
    owner = UserId.generate()
    updated_at = datetime(2026, 9, 1, 8, 30, tzinfo=timezone.utc)
    writer = _writer()
    writer.recall.return_value = [
        UserMemory(
            memory_id="0d6f8f2a-1a67-4f0b-9a0c-1f2f3a4b5c6d",
            user_id=owner,
            content="Reviews open tasks every Friday",
            topic="routine",
            created_at=updated_at,
            updated_at=updated_at,
        ),
        UserMemory(
            memory_id="7c1e2d3f-4a5b-4c6d-8e7f-9a0b1c2d3e4f",
            user_id=owner,
            content="Prefers morning deadlines",
            topic=None,
            created_at=updated_at,
            updated_at=updated_at,
        ),
    ]

    results = await _dispatcher(writer).dispatch(
        [LLMToolCall(name="recall", arguments={"query": "routine"}, call_id="c1")],
        owner_user_id=str(owner),
    )

    assert results[0].content == {
        "status": "ok",
        "count": 2,
        "memories": [
            {
                "content": "Reviews open tasks every Friday",
                "topic": "routine",
                "updated_at": "2026-09-01T08:30:00+00:00",
            },
            {
                "content": "Prefers morning deadlines",
                "topic": None,
                "updated_at": "2026-09-01T08:30:00+00:00",
            },
        ],
    }
    rendered = str(results[0].content)
    assert "0d6f8f2a" not in rendered
    assert str(owner) not in rendered


@pytest.mark.parametrize("arguments", [{}, {"query": None}])
async def test_recall_without_a_query_asks_for_the_freshest_entries(arguments: dict):
    owner = UserId.generate()
    writer = _writer()

    results = await _dispatcher(writer).dispatch(
        [LLMToolCall(name="recall", arguments=arguments, call_id="c1")],
        owner_user_id=str(owner),
    )

    assert results[0].content["status"] == "ok"
    writer.recall.assert_awaited_once_with(owner, query=None)


async def test_forget_requires_a_query():
    writer = _writer()

    results = await _dispatcher(writer).dispatch(
        [LLMToolCall(name="forget", arguments={}, call_id="c1")],
        owner_user_id=str(UserId.generate()),
    )

    assert results[0].content["status"] == "filter_error"
    assert results[0].content["data"]["message"] == "query is required"
    writer.forget.assert_not_awaited()


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("remember", {"content": "Prefers morning deadlines"}),
        ("recall", {"query": "deadlines"}),
        ("forget", {"query": "deadlines"}),
    ],
)
async def test_memory_tools_degrade_when_the_writer_is_not_wired(tool_name: str, arguments: dict):
    results = await _dispatcher(None).dispatch(
        [LLMToolCall(name=tool_name, arguments=arguments, call_id="c1")],
        owner_user_id=str(UserId.generate()),
    )

    assert results[0].name == tool_name
    assert results[0].content == {"error": "memory_unavailable"}


def test_executors_cover_exactly_the_three_memory_tools():
    assert set(build_memory_tool_executors(_writer())) == set(MEMORY_TOOL_NAMES)
    assert set(build_memory_tool_executors(None)) == {"remember", "recall", "forget"}


async def test_an_owner_id_that_is_not_a_user_id_never_reaches_the_vault():
    writer = _writer()

    results = await _dispatcher(writer).dispatch(
        [LLMToolCall(name="recall", arguments={}, call_id="c1")],
        owner_user_id="not-a-user-id",
    )

    assert results[0].content == {"error": "tool_execution_failed: ValueError"}
    writer.recall.assert_not_awaited()
