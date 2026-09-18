"""Render the compact long-term-memory pointer for the system prompt.

Vault content is never placed in the system message: that message is exempt from
history trimming, so anything in it is paid for on every turn. The pointer is a
small, size-capped table of contents instead: the entry count, the topic tags,
and the instruction to read the content through ``recall``.
"""

from __future__ import annotations

from typing import Optional, Sequence

_HEADER = "## Long-term memory"

_EMPTY = (
    _HEADER
    + "\n\nNothing is stored yet. When the user asks you to remember something or "
    "states a durable fact worth keeping, save it with remember."
)

# Caps that keep the pointer small whatever the size of the vault.
_MAX_TOPICS = 15
_MAX_TOPIC_CHARS = 40


class MemoryPointerRenderer:
    """Pure renderer: count and topics only, never the stored content."""

    @staticmethod
    def render(count: int, topics: Sequence[Optional[str]]) -> str:
        """Return the pointer block.

        ``topics`` may hold ``None`` or blank entries, because a topic is optional
        on a vault row; those are dropped. Duplicates collapse without regard to
        case, first-seen order is kept, long topics are truncated, and the list
        is capped.
        """
        if count <= 0:
            return _EMPTY

        seen: set[str] = set()
        rendered_topics: list[str] = []
        for topic in topics:
            if topic is None:
                continue
            cleaned = topic.strip()
            if not cleaned:
                continue
            if len(cleaned) > _MAX_TOPIC_CHARS:
                cleaned = cleaned[:_MAX_TOPIC_CHARS].rstrip() + "…"
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            rendered_topics.append(cleaned)
            if len(rendered_topics) >= _MAX_TOPICS:
                break

        lines = [f"{_HEADER}\n", f"Entries: {count}."]
        if rendered_topics:
            lines.append("Topics: " + ", ".join(rendered_topics) + ".")
        lines.append(
            "This is only a table of contents. Call recall whenever an answer may "
            "depend on what you already know about the user. Do not claim the "
            "memory is empty (the counter shows it is not), and do not restate its "
            "content without calling recall in this turn."
        )
        return "\n".join(lines)
