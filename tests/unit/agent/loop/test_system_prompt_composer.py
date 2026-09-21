"""Tests for the system prompt composer: block order and omission rules."""
from __future__ import annotations

import inspect
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.agent.application.loop.system_prompt_composer import SystemPromptComposer
from src.agent.domain.entities.assistant import Assistant
from src.agent.domain.entities.chat_agent_context import ChatAgentContext
from src.agent.domain.services.calendar_block_renderer import CalendarBlockRenderer
from src.agent.domain.services.conversation_state_renderer import ConversationStateRenderer
from src.agent.domain.services.memory_pointer_renderer import MemoryPointerRenderer
from src.agent.domain.value_objects.chat_id import ChatId
from src.shared.domain.value_objects.user_id import UserId

PROMPT = "You are a task assistant."
CALENDAR = "## Calendar\nNow: 2026-09-17 (Thursday), 14:05 in Europe/Berlin."
STATE = "## Conversation state\n- last_window: 2026-09-01 — 2026-09-17"
POINTER = "## Long-term memory\n\nEntries: 2."


def _assistant(system_prompt: str = PROMPT) -> Assistant:
    return Assistant(
        name="Task assistant",
        system_prompt=system_prompt,
        model_name="test-model",
        temperature=0.2,
    )


def test_system_message_carries_the_calendar_block_after_the_prompt():
    text = SystemPromptComposer().compose(_assistant(), CALENDAR, "", "")

    assert text == f"{PROMPT}\n\n{CALENDAR}"


def test_memory_pointer_is_appended_last():
    text = SystemPromptComposer().compose(_assistant(), CALENDAR, STATE, POINTER)

    assert text == f"{PROMPT}\n\n{CALENDAR}\n\n{STATE}\n\n{POINTER}"
    assert (
        text.index(PROMPT)
        < text.index("## Calendar")
        < text.index("## Conversation state")
        < text.index("## Long-term memory")
    )


def test_missing_pointer_leaves_the_system_message_clean():
    text = SystemPromptComposer().compose(_assistant(), CALENDAR, STATE, "")

    assert "Long-term memory" not in text
    assert text == f"{PROMPT}\n\n{CALENDAR}\n\n{STATE}"
    assert not text.endswith("\n")


def test_blank_conversation_state_leaves_no_empty_section():
    text = SystemPromptComposer().compose(_assistant(), CALENDAR, "  \n ", POINTER)

    assert "Conversation state" not in text
    assert text == f"{PROMPT}\n\n{CALENDAR}\n\n{POINTER}"
    assert "\n\n\n" not in text


def test_prompt_alone_when_every_block_is_blank():
    assert SystemPromptComposer().compose(_assistant(), "", "", "") == PROMPT


def test_blocks_are_trimmed_and_separated_by_exactly_one_blank_line():
    text = SystemPromptComposer().compose(
        _assistant(f"{PROMPT}\n\n\n"), f"\n{CALENDAR}\n", f"  {STATE}\n\n", f"\n\n{POINTER}  "
    )

    assert text == f"{PROMPT}\n\n{CALENDAR}\n\n{STATE}\n\n{POINTER}"


def test_composes_the_blocks_the_domain_renderers_produce():
    chat_id, user_id = ChatId.generate(), UserId.generate()
    context = ChatAgentContext.empty_for(chat_id, user_id)
    context.record_windowed_tool(
        tool="query_tasks",
        window_from="2026-09-01",
        window_to="2026-09-17",
        date_field="due",
        project="Home renovation",
    )
    calendar = CalendarBlockRenderer.render(
        datetime(2026, 9, 17, 12, 5, tzinfo=timezone.utc), ZoneInfo("Europe/Berlin")
    )
    state = ConversationStateRenderer.render(context)
    pointer = MemoryPointerRenderer.render(2, ["routine", "deadlines"])

    text = SystemPromptComposer().compose(_assistant(), calendar, state, pointer)

    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == ["## Calendar", "## Conversation state", "## Long-term memory"]
    assert text.startswith(PROMPT)
    assert "- today: 2026-09-17" in text
    assert "- last_project: Home renovation" in text
    assert "Entries: 2." in text


def test_empty_vault_pointer_is_still_a_block():
    pointer = MemoryPointerRenderer.render(0, [])

    text = SystemPromptComposer().compose(_assistant(), CALENDAR, "", pointer)

    assert text.endswith(pointer)
    assert "Nothing is stored yet" in text


def test_composer_has_no_input_through_which_memory_content_could_arrive():
    """It receives rendered blocks only: no repository, no vault entries."""
    parameters = list(inspect.signature(SystemPromptComposer.compose).parameters)

    assert parameters == [
        "self",
        "assistant",
        "calendar_block",
        "conversation_state_block",
        "memory_pointer",
    ]
