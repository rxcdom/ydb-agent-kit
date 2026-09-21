from __future__ import annotations

from datetime import tzinfo
from typing import Optional
from uuid import UUID

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.clock import UTC
from src.tasks.application.outcome import Outcome, TaskPresenter, ok
from src.tasks.application.task_lookup import TaskLookup
from src.tasks.domain.value_objects.local_calendar import LocalCalendar
from src.tasks.ports.repository_manager import TasksRepositoryManager


class DeleteTaskUseCase:
    """Removes one task of the owner for good.

    ``execute_by_id`` addresses the task by id. ``execute_by_reference``
    addresses it by human text and is the strictest write there is: it deletes
    only when the text identifies exactly one task, and refuses two matching
    tasks even when they look interchangeable.
    """

    def __init__(self, repositories: TasksRepositoryManager, timezone: tzinfo = UTC):
        self._repositories = repositories
        self._calendar = LocalCalendar(timezone)
        self._lookup = TaskLookup(repositories)

    async def execute_by_id(self, owner: UserId, task_id: UUID) -> None:
        task = await self._lookup.owned_task(owner, task_id)
        await self._repositories.tasks.delete(owner, task.task_id)

    async def execute_by_reference(
        self, owner: UserId, reference: str, project_scope: Optional[str] = None
    ) -> Outcome:
        projects = await self._repositories.projects.list_by_user(owner)
        presenter = TaskPresenter.for_projects(self._calendar, projects)

        task = await self._lookup.referenced_task(
            owner, reference, project_scope, projects, presenter
        )
        if isinstance(task, Outcome):
            return task

        await self._repositories.tasks.delete(owner, task.task_id)
        return ok({"deleted": presenter.summary(task)})
