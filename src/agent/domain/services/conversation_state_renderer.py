"""Render the ``## Conversation state`` block of the system prompt.

The block tells the model what the last windowed query in this chat looked like:
its window, date axis, project and tool. A follow-up question that names no
period can then reuse the previous one. Only known values are rendered; when
nothing is known the result is an empty string and the block is left out.
"""

from __future__ import annotations

from typing import Optional

from src.agent.domain.entities.chat_agent_context import ChatAgentContext

_HEADER = "## Conversation state"


class ConversationStateRenderer:
    """Pure renderer: known helper values only. No I/O, no randomness."""

    @staticmethod
    def render(context: Optional[ChatAgentContext]) -> str:
        """Return the conversation-state block, or ``""`` when there is no state."""
        if context is None or not context.has_context():
            return ""

        lines: list[str] = []

        window_from = context.last_window_from
        window_to = context.last_window_to
        if window_from and window_to:
            lines.append(f"- last_window: {window_from} — {window_to}")
        elif window_from:
            lines.append(f"- last_window: from {window_from}")
        elif window_to:
            lines.append(f"- last_window: until {window_to}")

        if context.last_date_field:
            lines.append(f"- last_date_field: {context.last_date_field}")

        if context.last_project:
            lines.append(f"- last_project: {context.last_project}")

        if context.last_tool:
            lines.append(f"- last_tool: {context.last_tool}")

        return "\n".join([_HEADER, *lines])
