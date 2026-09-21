from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, FrozenSet, List, Optional, Tuple
from uuid import UUID

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.domain.entities.task import Task
from src.tasks.domain.value_objects.date_axis import DateAxis
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus


@dataclass(frozen=True)
class TaskSearchCriteria:
    """What a windowed task query selects, always within one owner's rows.

    A query runs on one date axis and only sees tasks that have a value on it.
    The window is a half-open range of instants on that axis; a missing bound
    leaves its side open. Everything else is a narrowing filter, and ``None``
    means "do not filter on this".
    """

    axis: DateAxis = DateAxis.CREATED
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    project_id: Optional[UUID] = None
    statuses: Optional[FrozenSet[TaskStatus]] = None
    priorities: Optional[FrozenSet[TaskPriority]] = None
    text: Optional[str] = None

    @property
    def has_narrowing_filter(self) -> bool:
        return (
            self.project_id is not None
            or self.statuses is not None
            or self.priorities is not None
            or self.text is not None
        )

    def without_window(self) -> "TaskSearchCriteria":
        return replace(self, window_start=None, window_end=None)

    def without_filters(self) -> "TaskSearchCriteria":
        return replace(self, project_id=None, statuses=None, priorities=None, text=None)


@dataclass(frozen=True)
class ProjectActivity:
    """Aggregate of one owner's tasks sharing a project and a status.

    ``project_id`` is ``None`` for the unfiled tasks.
    """

    project_id: Optional[UUID]
    status: TaskStatus
    task_count: int
    first_created_at: datetime
    last_updated_at: datetime


class TaskRepository(ABC):
    """Owner-scoped storage of tasks. No method can reach another owner's row."""

    @abstractmethod
    async def save(self, task: Task, tx: Any = None) -> None:
        """Insert or replace the task, inside ``tx`` when one is given."""

    @abstractmethod
    async def find_by_id(self, user_id: UserId, task_id: UUID) -> Optional[Task]:
        """The owner's task with this id; ``None`` when the owner has no such task."""

    @abstractmethod
    async def delete(self, user_id: UserId, task_id: UUID) -> None:
        """Remove the owner's task for good. Deleting a missing task is not an error."""

    @abstractmethod
    async def list_by_user(self, user_id: UserId) -> List[Task]:
        """Every task of the owner, in every status, oldest first."""

    @abstractmethod
    async def list_by_project(self, user_id: UserId, project_id: UUID) -> List[Task]:
        """Every task the owner filed under one project, oldest first."""

    @abstractmethod
    async def search(
        self,
        user_id: UserId,
        criteria: TaskSearchCriteria,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
        descending: bool = False,
    ) -> List[Task]:
        """Matching tasks ordered by their instant on the criteria's axis."""

    @abstractmethod
    async def count(self, user_id: UserId, criteria: TaskSearchCriteria) -> int:
        """How many tasks ``search`` would return without paging."""

    @abstractmethod
    async def count_without_axis_date(self, user_id: UserId, criteria: TaskSearchCriteria) -> int:
        """Tasks that pass the narrowing filters but have no value on the axis.

        Such tasks cannot be placed in any window, so the window is ignored.
        """

    @abstractmethod
    async def date_bounds(
        self, user_id: UserId, axis: DateAxis
    ) -> Optional[Tuple[datetime, datetime]]:
        """Earliest and latest instant on the axis; ``None`` when no task has one."""

    @abstractmethod
    async def summarise_by_project(self, user_id: UserId) -> List[ProjectActivity]:
        """One aggregate per project and status present among the owner's tasks."""
