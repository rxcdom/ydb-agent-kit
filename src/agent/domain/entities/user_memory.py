"""One entry of the long-term memory vault (``user_memory`` table).

An entry is a free-form durable fact the user asked the agent to remember, or
one the agent judged worth keeping. Entries are read on demand by the ``recall``
tool; the system prompt only ever carries a compact pointer (count and topics).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.shared.domain.value_objects.user_id import UserId

# Long enough for a real fact, short enough that a full recall result stays
# cheap in tokens.
USER_MEMORY_CONTENT_MAX_LENGTH = 1000
USER_MEMORY_TOPIC_MAX_LENGTH = 100


@dataclass
class UserMemory:
    """One durable fact owned by ``user_id``."""

    memory_id: str
    user_id: UserId
    content: str
    topic: Optional[str]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not self.memory_id or not self.memory_id.strip():
            raise ValueError("memory_id must be non-empty")
        if not self.content or not self.content.strip():
            raise ValueError("content must be non-empty")
        if len(self.content) > USER_MEMORY_CONTENT_MAX_LENGTH:
            raise ValueError(
                f"content must be at most {USER_MEMORY_CONTENT_MAX_LENGTH} characters"
            )
        if self.topic is not None:
            if not self.topic.strip():
                self.topic = None
            elif len(self.topic) > USER_MEMORY_TOPIC_MAX_LENGTH:
                raise ValueError(
                    f"topic must be at most {USER_MEMORY_TOPIC_MAX_LENGTH} characters"
                )

    @staticmethod
    def generate_id() -> str:
        """Return a fresh UUID4 string for a new entry."""
        return str(uuid.uuid4())
