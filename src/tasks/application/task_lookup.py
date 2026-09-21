"""Finding the one row a caller means, by id or by human text.

The id-addressed face answers with domain errors; the text-addressed face
answers with outcomes. Both go through the same owner-scoped reads, and neither
ever returns a row of another owner.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union
from uuid import UUID

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.outcome import (
    Outcome,
    TaskPresenter,
    ambiguous_projects,
    ambiguous_tasks,
    project_not_found,
    task_not_found,
)
from src.tasks.domain.entities.project import Project
from src.tasks.domain.entities.task import Task
from src.tasks.domain.exceptions import ProjectNotFoundError, TaskNotFoundError
from src.tasks.domain.services.project_reference_resolver import ProjectReferenceResolver
from src.tasks.domain.services.reference_resolution import Ambiguous, Resolved
from src.tasks.domain.services.task_reference_resolver import TaskReferenceResolver
from src.tasks.ports.repository_manager import TasksRepositoryManager


class TaskLookup:
    def __init__(self, repositories: TasksRepositoryManager):
        self._repositories = repositories

    async def owned_task(self, owner: UserId, task_id: UUID) -> Task:
        """The owner's task, or ``TaskNotFoundError``.

        The read is owner-scoped already. The ownership check on top of it keeps
        the guarantee independent of any one repository implementation.
        """
        task = await self._repositories.tasks.find_by_id(owner, task_id)
        if task is None:
            raise TaskNotFoundError(f"Task {task_id} not found")
        task.ensure_owned_by(owner)
        return task

    async def owned_project(self, owner: UserId, project_id: UUID) -> Project:
        """The owner's project, or ``ProjectNotFoundError``."""
        project = await self._repositories.projects.find_by_id(owner, project_id)
        if project is None:
            raise ProjectNotFoundError(f"Project {project_id} not found")
        project.ensure_owned_by(owner)
        return project

    @staticmethod
    def referenced_project(
        projects: Sequence[Project], reference: str
    ) -> Union[Project, Outcome]:
        """The project a write refers to; ``ambiguous_source`` or ``not_found`` otherwise."""
        resolution = ProjectReferenceResolver.resolve(projects, reference)
        if isinstance(resolution, Resolved):
            return resolution.match
        if isinstance(resolution, Ambiguous):
            return ambiguous_projects(reference, resolution.candidates)
        return project_not_found(reference, projects)

    async def referenced_task(
        self,
        owner: UserId,
        reference: str,
        project_scope: Optional[str],
        projects: Sequence[Project],
        presenter: TaskPresenter,
    ) -> Union[Task, Outcome]:
        """The single task a write refers to, or the outcome that explains why there is none.

        Tasks of every status take part: reopening a finished task or deleting a
        cancelled one is as legitimate as editing an open one.
        """
        scope: Optional[Project] = None
        if project_scope is not None:
            resolved_scope = self.referenced_project(projects, project_scope)
            if isinstance(resolved_scope, Outcome):
                return resolved_scope
            scope = resolved_scope

        if scope is None:
            candidates = await self._repositories.tasks.list_by_user(owner)
        else:
            candidates = await self._repositories.tasks.list_by_project(owner, scope.project_id)

        resolution = TaskReferenceResolver.resolve(candidates, reference, scope)
        if isinstance(resolution, Resolved):
            resolution.match.ensure_owned_by(owner)
            return resolution.match
        if isinstance(resolution, Ambiguous):
            return ambiguous_tasks(reference, resolution.candidates, presenter)
        return task_not_found(reference, scope)
