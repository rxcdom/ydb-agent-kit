from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.domain.exceptions import InvalidTaskError, TaskAccessDeniedError
from src.tasks.domain.value_objects.date_axis import DateAxis
from src.tasks.domain.value_objects.local_calendar import to_supported_utc
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus

MAX_TASK_TITLE_LENGTH = 200
MAX_TASK_NOTES_LENGTH = 2000


@dataclass
class Task:
    """A unit of work owned by one user, optionally filed under a project.

    Invariant: ``status`` is ``DONE`` exactly when ``completed_at`` is set. The
    pair only changes through ``complete``, ``reopen`` and ``cancel``, and
    ``check_invariants`` refuses any other combination, both on construction
    and right before the entity is written.
    """

    task_id: UUID
    user_id: UserId
    project_id: Optional[UUID]
    title: str
    notes: Optional[str]
    status: TaskStatus
    priority: TaskPriority
    due_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        self.title = _clean_title(self.title)
        self.notes = _clean_notes(self.notes)
        self.status = _as_status(self.status)
        self.priority = _as_priority(self.priority)
        self.due_at = _clean_due_at(self.due_at)
        self.completed_at = _as_optional_utc(self.completed_at, "completed_at")
        self.created_at = _as_utc(self.created_at, "created_at")
        self.updated_at = _as_utc(self.updated_at, "updated_at")
        self.check_invariants()

    @classmethod
    def create(
        cls,
        user_id: UserId,
        title: str,
        now: datetime,
        project_id: Optional[UUID] = None,
        notes: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        due_at: Optional[datetime] = None,
    ) -> "Task":
        """A new open task."""
        return cls(
            task_id=uuid4(),
            user_id=user_id,
            project_id=project_id,
            title=title,
            notes=notes,
            status=TaskStatus.OPEN,
            priority=priority,
            due_at=due_at,
            completed_at=None,
            created_at=now,
            updated_at=now,
        )

    def check_invariants(self) -> None:
        is_done = self.status is TaskStatus.DONE
        if is_done and self.completed_at is None:
            raise InvalidTaskError("a done task must carry its completion instant")
        if not is_done and self.completed_at is not None:
            raise InvalidTaskError(
                f"only a done task carries a completion instant; status is {self.status.value}"
            )

    def ensure_owned_by(self, user_id: UserId) -> None:
        if self.user_id != user_id:
            raise TaskAccessDeniedError(f"Task {self.task_id} belongs to another owner")

    def complete(self, at: datetime) -> None:
        """Finish the task. A task that is already done keeps its first completion instant."""
        if self.status is TaskStatus.DONE:
            return
        self.completed_at = _as_utc(at, "completed_at")
        self.status = TaskStatus.DONE

    def reopen(self) -> None:
        self.status = TaskStatus.OPEN
        self.completed_at = None

    def cancel(self) -> None:
        self.status = TaskStatus.CANCELLED
        self.completed_at = None

    def change_status(self, status: TaskStatus, at: datetime) -> None:
        """Route a requested status through the transition that keeps the invariant."""
        target = _as_status(status)
        if target is TaskStatus.DONE:
            self.complete(at)
        elif target is TaskStatus.OPEN:
            self.reopen()
        else:
            self.cancel()

    def rename(self, title: str) -> None:
        self.title = _clean_title(title)

    def rewrite_notes(self, notes: Optional[str]) -> None:
        """Replace the notes; blank text clears them."""
        self.notes = _clean_notes(notes)

    def reprioritise(self, priority: TaskPriority) -> None:
        self.priority = _as_priority(priority)

    def reschedule(self, due_at: Optional[datetime]) -> None:
        """Set the due instant; ``None`` removes it."""
        self.due_at = _clean_due_at(due_at)

    def move_to_project(self, project_id: Optional[UUID]) -> None:
        """File the task under a project; ``None`` leaves it unfiled."""
        self.project_id = project_id

    def touch(self, at: datetime) -> None:
        self.updated_at = _as_utc(at, "updated_at")

    def date_on(self, axis: DateAxis) -> Optional[datetime]:
        """The instant this task has on a date axis; ``None`` when it has none."""
        if axis is DateAxis.DUE:
            return self.due_at
        if axis is DateAxis.COMPLETED:
            return self.completed_at
        return self.created_at


def _clean_title(title: str) -> str:
    cleaned = title.strip()
    if not cleaned:
        raise InvalidTaskError("title cannot be empty")
    if len(cleaned) > MAX_TASK_TITLE_LENGTH:
        raise InvalidTaskError(f"title cannot exceed {MAX_TASK_TITLE_LENGTH} characters")
    return cleaned


def _clean_notes(notes: Optional[str]) -> Optional[str]:
    cleaned = notes.strip() if notes is not None else ""
    if not cleaned:
        return None
    if len(cleaned) > MAX_TASK_NOTES_LENGTH:
        raise InvalidTaskError(f"notes cannot exceed {MAX_TASK_NOTES_LENGTH} characters")
    return cleaned


def _as_status(value: object) -> TaskStatus:
    try:
        return TaskStatus(value)
    except ValueError as error:
        raise InvalidTaskError(f"unknown task status: {value!r}") from error


def _as_priority(value: object) -> TaskPriority:
    try:
        return TaskPriority(value)
    except ValueError as error:
        raise InvalidTaskError(f"unknown task priority: {value!r}") from error


def _as_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise InvalidTaskError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _as_optional_utc(value: Optional[datetime], field_name: str) -> Optional[datetime]:
    return None if value is None else _as_utc(value, field_name)


def _clean_due_at(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        raise InvalidTaskError("due_at must be timezone-aware")
    due_at = to_supported_utc(value)
    if due_at is None:
        raise InvalidTaskError("due_at is outside the supported calendar range")
    return due_at
