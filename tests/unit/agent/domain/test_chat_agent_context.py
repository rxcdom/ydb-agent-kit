"""Tests for the per-chat conversation-state entity."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.agent.domain.entities.chat_agent_context import ChatAgentContext
from src.agent.domain.value_objects.chat_id import ChatId
from src.shared.domain.value_objects.user_id import UserId


def _context() -> ChatAgentContext:
    return ChatAgentContext.empty_for(ChatId.generate(), UserId.generate())


def test_empty_for_has_no_context_and_sets_timestamps():
    context = _context()

    assert context.has_context() is False
    assert context.created_at.tzinfo is not None
    assert context.created_at == context.updated_at
    assert context.last_tool is None
    assert context.last_window_from is None
    assert context.last_window_to is None
    assert context.last_date_field is None
    assert context.last_project is None


def test_record_windowed_tool_sets_fields_and_bumps_updated_at():
    context = _context()
    before = context.updated_at

    context.record_windowed_tool(
        tool="query_tasks",
        window_from="2026-05-01",
        window_to="2026-05-07",
        date_field="completed",
        project="Garden",
    )

    assert context.has_context() is True
    assert context.last_tool == "query_tasks"
    assert context.last_window_from == "2026-05-01"
    assert context.last_window_to == "2026-05-07"
    assert context.last_date_field == "completed"
    assert context.last_project == "Garden"
    assert context.updated_at >= before


def test_record_windowed_tool_normalises_blanks_to_none():
    context = _context()
    context.record_windowed_tool(
        tool="query_tasks",
        window_from="2026-05-01",
        window_to="2026-05-07",
        date_field="created",
        project="Garden",
    )

    context.record_windowed_tool(
        tool="  query_tasks  ",
        window_from="2026-06-01",
        window_to="2026-06-07",
        date_field="created",
        project="   ",
    )

    # A query without a project filter clears the previous project.
    assert context.last_project is None
    assert context.last_tool == "query_tasks"
    assert context.last_window_from == "2026-06-01"
    assert context.has_context() is True


def test_oversized_field_is_rejected_and_leaves_the_state_untouched():
    context = _context()
    context.record_windowed_tool(
        tool="query_tasks",
        window_from="2026-05-01",
        window_to="2026-05-07",
        date_field="created",
        project="Garden",
    )

    with pytest.raises(ValueError):
        context.record_windowed_tool(
            tool="query_tasks",
            window_from="2026-06-01",
            window_to="2026-06-07",
            date_field="created",
            project="x" * 65,
        )

    assert context.last_window_from == "2026-05-01"
    assert context.last_project == "Garden"


def test_constructor_normalises_and_validates_stored_values():
    now = datetime.now(timezone.utc)
    context = ChatAgentContext(
        chat_id=ChatId.generate(),
        user_id=UserId.generate(),
        created_at=now,
        updated_at=now,
        last_tool=" query_tasks ",
        last_project="",
    )
    assert context.last_tool == "query_tasks"
    assert context.last_project is None

    with pytest.raises(ValueError):
        ChatAgentContext(
            chat_id=ChatId.generate(),
            user_id=UserId.generate(),
            created_at=now,
            updated_at=now,
            last_tool=42,  # type: ignore[arg-type]
        )
