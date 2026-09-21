"""Capture the conversation state of a chat from the agent's own tool trace.

The loop already records every executed tool call with its arguments and its
result. After a turn this service reads that trace, takes the last windowed
query that succeeded, and stores its resolved window, date axis and project as
the chat's ``ChatAgentContext``. The next turn renders it as the
``## Conversation state`` block, so a follow-up such as "and in Kitchen?" can
reuse the window instead of guessing a new one.

Only an ``ok`` result carries a resolved window. Every other outcome (a coverage
gap, an empty filter, an ambiguous reference, an argument error) is skipped, so a
failed probe never wipes the state a good earlier query left behind.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from src.agent.application.tools.schemas import QUERY_TASKS
from src.agent.domain.entities.chat_agent_context import ChatAgentContext
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.ports.llm_tooling import AgentRequestDebugTrace
from src.agent.ports.repository_manager import RepositoryManager
from src.shared.domain.value_objects.user_id import UserId

logger = logging.getLogger(__name__)

# Tools that resolve a date window and answer within it.
WINDOWED_TOOLS = frozenset({QUERY_TASKS})

_OK_STATUS = "ok"


@dataclass(frozen=True)
class _Capture:
    tool: str
    window_from: Optional[str]
    window_to: Optional[str]
    date_field: Optional[str]
    project: Optional[str]


def _text(value: Any) -> Optional[str]:
    """Return ``value`` when it is a non-blank string, else ``None``."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _extract_last_capture(trace: AgentRequestDebugTrace) -> Optional[_Capture]:
    """Return the last windowed tool call of the trace that resolved a window.

    A tool result is data produced outside this module, so every value is
    type-checked before it is used and anything unexpected counts as absent.
    """
    last: Optional[_Capture] = None
    for iteration in trace.request_flow:
        for call in iteration.function_calls:
            if call.name not in WINDOWED_TOOLS:
                continue
            result = _mapping(call.result)
            if result.get("status") != _OK_STATUS:
                continue
            data = _mapping(result.get("data"))
            window = _mapping(data.get("window_used"))
            window_from = _text(window.get("from"))
            window_to = _text(window.get("to"))
            if window_from is None and window_to is None:
                continue
            last = _Capture(
                tool=call.name,
                window_from=window_from,
                window_to=window_to,
                date_field=_text(data.get("date_field")),
                # Present only when the query was narrowed to a single project.
                project=_text(_mapping(data.get("project_resolved")).get("name")),
            )
    return last


class ConversationStateWriter:
    """Store the last successful windowed query of a turn as the chat's state."""

    def __init__(self, repository_manager: RepositoryManager) -> None:
        self._repositories = repository_manager

    async def record_turn(
        self, *, chat_id: ChatId, user_id: UserId, trace: AgentRequestDebugTrace
    ) -> None:
        """Upsert the chat's conversation state from the trace of one turn.

        A turn without a successful windowed query changes nothing: the state
        recorded earlier, if any, stays as it is.
        """
        capture = _extract_last_capture(trace)
        if capture is None:
            return

        existing = await self._repositories.chat_agent_context.find_by_chat_id(chat_id)
        context = existing or ChatAgentContext.empty_for(chat_id, user_id)
        try:
            context.record_windowed_tool(
                tool=capture.tool,
                window_from=capture.window_from,
                window_to=capture.window_to,
                date_field=capture.date_field,
                project=capture.project,
            )
        except ValueError:
            # The entity refuses values it cannot hold (for example an oversized
            # one). The state is a convenience, so such a capture is dropped and
            # the previous state stays in place.
            logger.warning(
                "Conversation state not recorded for chat %s: captured values were rejected",
                chat_id,
                exc_info=True,
            )
            return

        await self._repositories.chat_agent_context.save(context)
        logger.info(
            "Conversation state recorded for chat %s: tool=%s window=%s..%s date_field=%s "
            "project_present=%s",
            chat_id,
            context.last_tool,
            context.last_window_from,
            context.last_window_to,
            context.last_date_field,
            context.last_project is not None,
        )
