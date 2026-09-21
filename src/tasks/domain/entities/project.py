from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.domain.exceptions import InvalidProjectError, ProjectAccessDeniedError

MAX_PROJECT_NAME_LENGTH = 60
MAX_PROJECT_DESCRIPTION_LENGTH = 200


@dataclass
class Project:
    """A named group of tasks owned by one user.

    Names are for people and may collide, within one owner as well; identity is
    the id. That is why a textual reference to a project has to be resolved and
    can turn out ambiguous.
    """

    project_id: UUID
    user_id: UserId
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        self.name = _clean_name(self.name)
        self.description = _clean_description(self.description)
        self.created_at = _as_utc(self.created_at, "created_at")
        self.updated_at = _as_utc(self.updated_at, "updated_at")

    @classmethod
    def create(
        cls, user_id: UserId, name: str, description: Optional[str], now: datetime
    ) -> "Project":
        return cls(
            project_id=uuid4(),
            user_id=user_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )

    def ensure_owned_by(self, user_id: UserId) -> None:
        if self.user_id != user_id:
            raise ProjectAccessDeniedError(f"Project {self.project_id} belongs to another owner")


def _clean_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise InvalidProjectError("name cannot be empty")
    if len(cleaned) > MAX_PROJECT_NAME_LENGTH:
        raise InvalidProjectError(f"name cannot exceed {MAX_PROJECT_NAME_LENGTH} characters")
    return cleaned


def _clean_description(description: Optional[str]) -> Optional[str]:
    cleaned = description.strip() if description is not None else ""
    if not cleaned:
        return None
    if len(cleaned) > MAX_PROJECT_DESCRIPTION_LENGTH:
        raise InvalidProjectError(
            f"description cannot exceed {MAX_PROJECT_DESCRIPTION_LENGTH} characters"
        )
    return cleaned


def _as_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise InvalidProjectError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)
