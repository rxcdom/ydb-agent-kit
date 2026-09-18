from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.agent.domain.value_objects.chat_id import ChatId
from src.shared.domain.value_objects.user_id import UserId

CHAT_TITLE_MAX_LENGTH = 200


@dataclass
class Chat:
    """A conversation owned by one user."""

    chat_id: ChatId
    user_id: UserId
    created_at: datetime
    updated_at: datetime
    title: Optional[str] = None

    def __post_init__(self) -> None:
        if self.title is not None:
            stripped = self.title.strip()
            if len(stripped) > CHAT_TITLE_MAX_LENGTH:
                raise ValueError(
                    f"Chat title must be at most {CHAT_TITLE_MAX_LENGTH} characters"
                )
            # A blank title carries no information, so it is stored as absent.
            self.title = stripped or None
