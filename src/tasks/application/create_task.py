from __future__ import annotations

from datetime import date, datetime, tzinfo
from typing import Optional, Union
from uuid import UUID

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.arguments import due_instant, read_choice, read_day
from src.tasks.application.clock import UTC, Clock, utc_now
from src.tasks.application.outcome import INVALID_TASK, Outcome, TaskPresenter, filter_error, ok
from src.tasks.application.task_lookup import TaskLookup
from src.tasks.domain.entities.task import Task
from src.tasks.domain.exceptions import InvalidTaskError
from src.tasks.domain.value_objects.local_calendar import LocalCalendar
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.ports.repository_manager import TasksRepositoryManager


class CreateTaskUseCase:
    """Creates an open task for the owner.

    ``execute_by_id`` names the project by id and answers with the task or a
    domain error. ``execute_by_reference`` names it by human text and answers
    with an outcome; it files nothing under a project it could not identify
    beyond doubt. Both build and store the task through the same code.
    """

    def __init__(
        self,
        repositories: TasksRepositoryManager,
        timezone: tzinfo = UTC,
        clock: Clock = utc_now,
    ):
        self._repositories = repositories
        self._calendar = LocalCalendar(timezone)
        self._clock = clock
        self._lookup = TaskLookup(repositories)

    async def execute_by_id(
        self,
        owner: UserId,
        title: str,
        *,
        project_id: Optional[UUID] = None,
        due_at: Union[date, datetime, None] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        notes: Optional[str] = None,
    ) -> Task:
        if project_id is not None:
            await self._lookup.owned_project(owner, project_id)
        return await self._create(owner, title, project_id, due_at, priority, notes)

    async def execute_by_reference(
        self,
        owner: UserId,
        title: str,
        *,
        project: Optional[str] = None,
        due_at: Optional[str] = None,
        priority: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Outcome:
        chosen_priority = (
            TaskPriority.NORMAL if priority is None
            else read_choice(TaskPriority, priority, "priority")
        )
        if isinstance(chosen_priority, Outcome):
            return chosen_priority

        due_day = None if due_at is None else read_day(due_at, "due_at")
        if isinstance(due_day, Outcome):
            return due_day

        projects = await self._repositories.projects.list_by_user(owner)
        project_id: Optional[UUID] = None
        if project is not None:
            resolved = TaskLookup.referenced_project(projects, project)
            if isinstance(resolved, Outcome):
                return resolved
            project_id = resolved.project_id

        try:
            task = await self._create(owner, title, project_id, due_day, chosen_priority, notes)
        except InvalidTaskError as error:
            return filter_error(INVALID_TASK, f"The task was not created: {error}.")

        presenter = TaskPresenter.for_projects(self._calendar, projects)
        return ok({"created": presenter.summary(task)})

    async def _create(
        self,
        owner: UserId,
        title: str,
        project_id: Optional[UUID],
        due: Union[date, datetime, None],
        priority: TaskPriority,
        notes: Optional[str],
    ) -> Task:
        task = Task.create(
            user_id=owner,
            title=title,
            now=self._clock(),
            project_id=project_id,
            notes=notes,
            priority=priority,
            due_at=None if due is None else due_instant(self._calendar, due),
        )
        await self._repositories.tasks.save(task)
        return task
