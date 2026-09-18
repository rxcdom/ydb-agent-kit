from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.accounts.domain.exceptions import InvalidUserError
from src.shared.domain.value_objects.user_id import UserId

MAX_DISPLAY_NAME_LENGTH = 80


@dataclass
class User:
    """A user of the demo. Its id doubles as its bearer credential."""

    user_id: UserId
    display_name: Optional[str]
    created_at: datetime

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise InvalidUserError("created_at must be timezone-aware")
        if self.display_name is not None:
            self.display_name = self.display_name.strip() or None
        if self.display_name is not None and len(self.display_name) > MAX_DISPLAY_NAME_LENGTH:
            raise InvalidUserError(
                f"display_name cannot exceed {MAX_DISPLAY_NAME_LENGTH} characters"
            )
