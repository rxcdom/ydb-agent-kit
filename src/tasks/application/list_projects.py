from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.clock import UTC
from src.tasks.application.outcome import Outcome, no_data, ok
from src.tasks.domain.entities.project import Project
from src.tasks.domain.value_objects.local_calendar import LocalCalendar
from src.tasks.domain.value_objects.task_status import TaskStatus
from src.tasks.ports.repository_manager import TasksRepositoryManager
from src.tasks.ports.task_repository import ProjectActivity


@dataclass(frozen=True)
class ActivityTotals:
    """Task counts and the span of activity of one project, or of the unfiled tasks."""

    open_count: int = 0
    done_count: int = 0
    first_activity: Optional[datetime] = None
    last_activity: Optional[datetime] = None

    @classmethod
    def of(cls, rows: Iterable[ProjectActivity]) -> "ActivityTotals":
        """Fold per-status aggregates. Tasks of every status count as activity."""
        rows = list(rows)
        if not rows:
            return cls()
        return cls(
            open_count=sum(row.task_count for row in rows if row.status is TaskStatus.OPEN),
            done_count=sum(row.task_count for row in rows if row.status is TaskStatus.DONE),
            first_activity=min(row.first_created_at for row in rows),
            last_activity=max(row.last_updated_at for row in rows),
        )


@dataclass(frozen=True)
class ProjectOverview:
    project: Project
    totals: ActivityTotals


@dataclass(frozen=True)
class ProjectInventory:
    projects: List[ProjectOverview]
    unfiled: ActivityTotals
    has_tasks: bool


class ListProjectsUseCase:
    """The owner's projects with what is in them.

    ``list_overviews`` serves the id-addressed API. ``execute`` serves the
    text-addressed face: the same inventory as an outcome, with names and local
    days instead of ids and instants.
    """

    def __init__(self, repositories: TasksRepositoryManager, timezone: tzinfo = UTC):
        self._repositories = repositories
        self._calendar = LocalCalendar(timezone)

    async def list_overviews(self, owner: UserId) -> List[ProjectOverview]:
        return (await self._inventory(owner)).projects

    async def execute(self, owner: UserId) -> Outcome:
        inventory = await self._inventory(owner)
        if not inventory.projects and not inventory.has_tasks:
            return no_data(projects_count=0)
        return ok(
            {
                "projects": [
                    {
                        "name": overview.project.name,
                        "description": overview.project.description,
                        **self._render_totals(overview.totals),
                    }
                    for overview in inventory.projects
                ],
                "unfiled": self._render_totals(inventory.unfiled),
            }
        )

    async def _inventory(self, owner: UserId) -> ProjectInventory:
        projects = await self._repositories.projects.list_by_user(owner)
        activity = await self._repositories.tasks.summarise_by_project(owner)

        by_project: Dict[Optional[UUID], List[ProjectActivity]] = {}
        for row in activity:
            by_project.setdefault(row.project_id, []).append(row)

        return ProjectInventory(
            projects=[
                ProjectOverview(project, ActivityTotals.of(by_project.get(project.project_id, [])))
                for project in projects
            ],
            unfiled=ActivityTotals.of(by_project.get(None, [])),
            has_tasks=bool(activity),
        )

    def _render_totals(self, totals: ActivityTotals) -> Dict[str, Any]:
        def day(instant: Optional[datetime]) -> Optional[str]:
            return None if instant is None else self._calendar.day_of(instant).isoformat()

        return {
            "open_count": totals.open_count,
            "done_count": totals.done_count,
            "first_activity": day(totals.first_activity),
            "last_activity": day(totals.last_activity),
        }
