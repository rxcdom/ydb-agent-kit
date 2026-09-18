"""Per-chat helper state the agent carries from one turn to the next.

The tool loop is stateless per request and only the final assistant text is
stored as a message, so the window, the date axis and the project of the last
query would be lost between turns. This entity keeps that small record per chat.
It is rendered into the system prompt as the ``## Conversation state`` block, so
a follow-up question can reuse the previous window instead of guessing one.

It is derived state: it can always be rebuilt from the next successful query.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.agent.domain.value_objects.chat_id import ChatId
from src.shared.domain.value_objects.user_id import UserId

# Values come from controlled sources (ISO dates, the date-axis vocabulary, tool
# names, project names), but unbounded input is never trusted.
_MAX_FIELD_LENGTH = 64


def _normalise(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected str or None, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) > _MAX_FIELD_LENGTH:
        raise ValueError(
            f"Conversation-state field exceeds {_MAX_FIELD_LENGTH} characters"
        )
    return stripped


@dataclass
class ChatAgentContext:
    """Window, date axis, project and tool of the last windowed query in a chat."""

    chat_id: ChatId
    user_id: UserId
    created_at: datetime
    updated_at: datetime
    last_tool: Optional[str] = None
    last_window_from: Optional[str] = None
    last_window_to: Optional[str] = None
    last_date_field: Optional[str] = None
    last_project: Optional[str] = None

    def __post_init__(self) -> None:
        self.last_tool = _normalise(self.last_tool)
        self.last_window_from = _normalise(self.last_window_from)
        self.last_window_to = _normalise(self.last_window_to)
        self.last_date_field = _normalise(self.last_date_field)
        self.last_project = _normalise(self.last_project)

    @classmethod
    def empty_for(cls, chat_id: ChatId, user_id: UserId) -> ChatAgentContext:
        """Build a context with no helper values and both timestamps set to now."""
        now = datetime.now(timezone.utc)
        return cls(chat_id=chat_id, user_id=user_id, created_at=now, updated_at=now)

    def record_windowed_tool(
        self,
        *,
        tool: str,
        window_from: Optional[str],
        window_to: Optional[str],
        date_field: Optional[str],
        project: Optional[str],
    ) -> None:
        """Overwrite the state with the latest windowed query.

        The window is the one the tool actually resolved, not the one the model
        asked for. Every value is replaced, so a query without a project filter
        clears ``last_project``: no project filter is active any more.
        """
        tool_name = _normalise(tool)
        resolved_from = _normalise(window_from)
        resolved_to = _normalise(window_to)
        axis = _normalise(date_field)
        project_name = _normalise(project)

        self.last_tool = tool_name
        self.last_window_from = resolved_from
        self.last_window_to = resolved_to
        self.last_date_field = axis
        self.last_project = project_name
        self.updated_at = datetime.now(timezone.utc)

    def has_context(self) -> bool:
        """Return True when at least one helper value is known."""
        return any(
            (
                self.last_tool,
                self.last_window_from,
                self.last_window_to,
                self.last_date_field,
                self.last_project,
            )
        )
