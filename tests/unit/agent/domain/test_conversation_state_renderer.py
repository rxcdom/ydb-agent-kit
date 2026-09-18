"""Tests for the conversation-state block of the system prompt."""
from __future__ import annotations

from src.agent.domain.entities.chat_agent_context import ChatAgentContext
from src.agent.domain.services.conversation_state_renderer import ConversationStateRenderer
from src.agent.domain.value_objects.chat_id import ChatId
from src.shared.domain.value_objects.user_id import UserId


def _context() -> ChatAgentContext:
    return ChatAgentContext.empty_for(ChatId.generate(), UserId.generate())


def test_none_or_empty_renders_empty_string():
    assert ConversationStateRenderer.render(None) == ""
    assert ConversationStateRenderer.render(_context()) == ""


def test_full_context_renders_the_documented_block():
    context = _context()
    context.record_windowed_tool(
        tool="query_tasks",
        window_from="2026-09-01",
        window_to="2026-09-17",
        date_field="created",
        project="Home renovation",
    )

    assert ConversationStateRenderer.render(context) == (
        "## Conversation state\n"
        "- last_window: 2026-09-01 — 2026-09-17\n"
        "- last_date_field: created\n"
        "- last_project: Home renovation\n"
        "- last_tool: query_tasks"
    )


def test_absent_values_are_omitted():
    context = _context()
    context.record_windowed_tool(
        tool="query_tasks",
        window_from="2026-05-01",
        window_to="2026-05-07",
        date_field="due",
        project=None,
    )
    block = ConversationStateRenderer.render(context)

    assert "- last_window: 2026-05-01 — 2026-05-07" in block
    assert "- last_date_field: due" in block
    assert "last_project" not in block
    assert "- last_tool: query_tasks" in block


def test_open_ended_window_phrasing():
    only_from = _context()
    only_from.last_window_from = "2026-05-01"
    assert "- last_window: from 2026-05-01" in ConversationStateRenderer.render(only_from)

    only_to = _context()
    only_to.last_window_to = "2026-05-07"
    assert "- last_window: until 2026-05-07" in ConversationStateRenderer.render(only_to)
