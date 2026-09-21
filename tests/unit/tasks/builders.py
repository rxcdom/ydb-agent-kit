"""Small builders that keep the tests about behaviour, not about constructor noise."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.domain.entities.project import Project
from src.tasks.domain.entities.task import Task
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus

UTC = timezone.utc
NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


def at(year: int, month: int, day: int, hour: int = 12, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def a_project(
    owner: UserId, name: str, description: Optional[str] = None, created_at: datetime = NOW
) -> Project:
    return Project(
        project_id=uuid4(),
        user_id=owner,
        name=name,
        description=description,
        created_at=created_at,
        updated_at=created_at,
    )


def a_task(
    owner: UserId,
    title: str,
    *,
    project: Optional[Project] = None,
    project_id: Optional[UUID] = None,
    status: TaskStatus = TaskStatus.OPEN,
    priority: TaskPriority = TaskPriority.NORMAL,
    created_at: datetime = NOW,
    due_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    notes: Optional[str] = None,
) -> Task:
    if status is TaskStatus.DONE and completed_at is None:
        completed_at = created_at
    return Task(
        task_id=uuid4(),
        user_id=owner,
        project_id=project.project_id if project is not None else project_id,
        title=title,
        notes=notes,
        status=status,
        priority=priority,
        due_at=due_at,
        completed_at=completed_at,
        created_at=created_at,
        updated_at=completed_at or created_at,
    )
