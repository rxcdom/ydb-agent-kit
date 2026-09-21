from __future__ import annotations

from typing import Optional

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.clock import Clock, utc_now
from src.tasks.domain.entities.project import Project
from src.tasks.ports.repository_manager import TasksRepositoryManager


class CreateProjectUseCase:
    """Creates a project for the owner.

    Projects have an id-addressed face only: the agent files tasks under
    projects but never creates one, which keeps its write surface on tasks.
    """

    def __init__(self, repositories: TasksRepositoryManager, clock: Clock = utc_now):
        self._repositories = repositories
        self._clock = clock

    async def execute(
        self, owner: UserId, name: str, description: Optional[str] = None
    ) -> Project:
        project = Project.create(owner, name, description, self._clock())
        await self._repositories.projects.save(project)
        return project
