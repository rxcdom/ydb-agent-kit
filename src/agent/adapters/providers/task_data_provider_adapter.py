"""Implements the agent's ``TaskDataProvider`` port on top of the tasks use cases.

This is the one place where the agent module touches the tasks module. The
agent's application layer sees only the port and plain dictionaries; the tasks
module does not know the agent exists. The adapter translates in one direction
only: it hands the owner id and the human-text arguments to the text-addressed
face of each use case and returns the outcome as an envelope.
"""
from __future__ import annotations

from src.agent.ports.task_data_provider import TaskDataProvider
from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.create_task import CreateTaskUseCase
from src.tasks.application.delete_task import DeleteTaskUseCase
from src.tasks.application.list_projects import ListProjectsUseCase
from src.tasks.application.query_tasks import QueryTasksUseCase
from src.tasks.application.update_task import TaskChangeRequest, UpdateTaskUseCase


class TaskDataProviderAdapter(TaskDataProvider):
    def __init__(
        self,
        list_projects: ListProjectsUseCase,
        query_tasks: QueryTasksUseCase,
        create_task: CreateTaskUseCase,
        update_task: UpdateTaskUseCase,
        delete_task: DeleteTaskUseCase,
    ) -> None:
        self._list_projects = list_projects
        self._query_tasks = query_tasks
        self._create_task = create_task
        self._update_task = update_task
        self._delete_task = delete_task

    async def list_projects(self, *, owner_user_id: str) -> dict:
        outcome = await self._list_projects.execute(UserId.from_string(owner_user_id))
        return outcome.to_envelope()

    async def query_tasks(
        self,
        *,
        owner_user_id: str,
        date_from: str | None,
        date_to: str | None,
        date_field: str,
        project: str | None,
        statuses: list[str] | None,
        priority: list[str] | None,
        text: str | None,
        group_by: str,
        view: str,
    ) -> dict:
        outcome = await self._query_tasks.execute(
            UserId.from_string(owner_user_id),
            date_from=date_from,
            date_to=date_to,
            date_field=date_field,
            project=project,
            statuses=statuses,
            priority=priority,
            text=text,
            group_by=group_by,
            view=view,
        )
        return outcome.to_envelope()

    async def create_task(
        self,
        *,
        owner_user_id: str,
        title: str,
        project: str | None,
        due_at: str | None,
        priority: str | None,
        notes: str | None,
    ) -> dict:
        outcome = await self._create_task.execute_by_reference(
            UserId.from_string(owner_user_id),
            title,
            project=project,
            due_at=due_at,
            priority=priority,
            notes=notes,
        )
        return outcome.to_envelope()

    async def update_task(
        self,
        *,
        owner_user_id: str,
        task: str,
        project_scope: str | None,
        set_status: str | None,
        set_title: str | None,
        set_due_at: str | None,
        set_priority: str | None,
        set_project: str | None,
        set_notes: str | None,
    ) -> dict:
        outcome = await self._update_task.execute_by_reference(
            UserId.from_string(owner_user_id),
            task,
            project_scope,
            TaskChangeRequest(
                set_status=set_status,
                set_title=set_title,
                set_due_at=set_due_at,
                set_priority=set_priority,
                set_project=set_project,
                set_notes=set_notes,
            ),
        )
        return outcome.to_envelope()

    async def delete_task(
        self, *, owner_user_id: str, task: str, project_scope: str | None
    ) -> dict:
        outcome = await self._delete_task.execute_by_reference(
            UserId.from_string(owner_user_id), task, project_scope
        )
        return outcome.to_envelope()
