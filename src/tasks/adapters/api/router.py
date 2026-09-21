"""HTTP face of the tasks module: projects and tasks addressed by id.

Handlers take the owner from the authenticated principal, never from the path
or the body, and contain no error handling: use cases raise domain errors and
the gateway turns them into responses in one place.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional, Union
from uuid import UUID

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.accounts.adapters.api.auth import get_authorized_user
from src.shared.domain.dtos.authorized_user import AuthorizedUser
from src.tasks.application.create_project import CreateProjectUseCase
from src.tasks.application.create_task import CreateTaskUseCase
from src.tasks.application.delete_task import DeleteTaskUseCase
from src.tasks.application.list_projects import ListProjectsUseCase, ProjectOverview
from src.tasks.application.query_tasks import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    QueryTasksUseCase,
)
from src.tasks.application.update_task import TaskChanges, UpdateTaskUseCase
from src.tasks.domain.entities.project import (
    MAX_PROJECT_DESCRIPTION_LENGTH,
    MAX_PROJECT_NAME_LENGTH,
    Project,
)
from src.tasks.domain.entities.task import MAX_TASK_NOTES_LENGTH, MAX_TASK_TITLE_LENGTH, Task
from src.tasks.domain.value_objects.date_axis import DateAxis
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus

projects_router = APIRouter(prefix="/projects", tags=["projects"])
tasks_router = APIRouter(prefix="/tasks", tags=["tasks"])

# A bare day is due at the local midnight that starts it; a moment is kept as given.
DueAt = Union[date, datetime]


class CreateProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=MAX_PROJECT_NAME_LENGTH)
    description: Optional[str] = Field(default=None, max_length=MAX_PROJECT_DESCRIPTION_LENGTH)


class ProjectResponse(BaseModel):
    project_id: UUID
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, project: Project) -> "ProjectResponse":
        return cls(
            project_id=project.project_id,
            name=project.name,
            description=project.description,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )


class ProjectOverviewResponse(ProjectResponse):
    open_count: int
    done_count: int

    @classmethod
    def of_overview(cls, overview: ProjectOverview) -> "ProjectOverviewResponse":
        return cls(
            **ProjectResponse.of(overview.project).model_dump(),
            open_count=overview.totals.open_count,
            done_count=overview.totals.done_count,
        )


class ListProjectsResponse(BaseModel):
    projects: List[ProjectOverviewResponse]


class CreateTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=MAX_TASK_TITLE_LENGTH)
    project_id: Optional[UUID] = None
    due_at: Optional[DueAt] = None
    priority: TaskPriority = TaskPriority.NORMAL
    notes: Optional[str] = Field(default=None, max_length=MAX_TASK_NOTES_LENGTH)


class UpdateTaskRequest(BaseModel):
    """Any subset of the editable fields.

    A field left out is not touched. ``null`` is a value for the fields that can
    be empty: it removes the due date, unfiles the task, or clears the notes.
    """

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(default=None, min_length=1, max_length=MAX_TASK_TITLE_LENGTH)
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    due_at: Optional[DueAt] = None
    project_id: Optional[UUID] = None
    notes: Optional[str] = Field(default=None, max_length=MAX_TASK_NOTES_LENGTH)

    @model_validator(mode="after")
    def _required_fields_are_not_null(self) -> "UpdateTaskRequest":
        for name in ("title", "status", "priority"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self

    def to_changes(self) -> TaskChanges:
        """Only the fields present in the body, so "absent" and "null" stay apart."""
        return TaskChanges(**{name: getattr(self, name) for name in self.model_fields_set})


class TaskResponse(BaseModel):
    task_id: UUID
    project_id: Optional[UUID]
    title: str
    notes: Optional[str]
    status: TaskStatus
    priority: TaskPriority
    due_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, task: Task) -> "TaskResponse":
        return cls(
            task_id=task.task_id,
            project_id=task.project_id,
            title=task.title,
            notes=task.notes,
            status=task.status,
            priority=task.priority,
            due_at=task.due_at,
            completed_at=task.completed_at,
            created_at=task.created_at,
            updated_at=task.updated_at,
        )


class ListTasksResponse(BaseModel):
    tasks: List[TaskResponse]
    total_count: int
    has_more: bool


@projects_router.post("", status_code=status.HTTP_201_CREATED, response_model=ProjectResponse)
@inject
async def create_project(
    request: CreateProjectRequest,
    user: AuthorizedUser = Depends(get_authorized_user),
    use_case: CreateProjectUseCase = Depends(Provide["tasks.create_project_use_case"]),
) -> ProjectResponse:
    project = await use_case.execute(user.user_id, request.name, request.description)
    return ProjectResponse.of(project)


@projects_router.get("", response_model=ListProjectsResponse)
@inject
async def list_projects(
    user: AuthorizedUser = Depends(get_authorized_user),
    use_case: ListProjectsUseCase = Depends(Provide["tasks.list_projects_use_case"]),
) -> ListProjectsResponse:
    overviews = await use_case.list_overviews(user.user_id)
    return ListProjectsResponse(
        projects=[ProjectOverviewResponse.of_overview(overview) for overview in overviews]
    )


@tasks_router.post("", status_code=status.HTTP_201_CREATED, response_model=TaskResponse)
@inject
async def create_task(
    request: CreateTaskRequest,
    user: AuthorizedUser = Depends(get_authorized_user),
    use_case: CreateTaskUseCase = Depends(Provide["tasks.create_task_use_case"]),
) -> TaskResponse:
    task = await use_case.execute_by_id(
        user.user_id,
        request.title,
        project_id=request.project_id,
        due_at=request.due_at,
        priority=request.priority,
        notes=request.notes,
    )
    return TaskResponse.of(task)


@tasks_router.get("", response_model=ListTasksResponse)
@inject
async def list_tasks(
    status_filter: Optional[TaskStatus] = Query(default=None, alias="status"),
    project_id: Optional[UUID] = Query(default=None),
    date_field: DateAxis = Query(
        default=DateAxis.CREATED,
        description="The date the window and the order apply to. Tasks without a value "
        "on it are not listed.",
    ),
    date_from: Optional[date] = Query(default=None, description="First local day, inclusive."),
    date_to: Optional[date] = Query(default=None, description="Last local day, inclusive."),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    user: AuthorizedUser = Depends(get_authorized_user),
    use_case: QueryTasksUseCase = Depends(Provide["tasks.query_tasks_use_case"]),
) -> ListTasksResponse:
    page = await use_case.list_page(
        user.user_id,
        status=status_filter,
        project_id=project_id,
        date_field=date_field,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )
    return ListTasksResponse(
        tasks=[TaskResponse.of(task) for task in page.tasks],
        total_count=page.total_count,
        has_more=page.has_more,
    )


@tasks_router.patch("/{task_id}", response_model=TaskResponse)
@inject
async def update_task(
    task_id: UUID,
    request: UpdateTaskRequest,
    user: AuthorizedUser = Depends(get_authorized_user),
    use_case: UpdateTaskUseCase = Depends(Provide["tasks.update_task_use_case"]),
) -> TaskResponse:
    task = await use_case.execute_by_id(user.user_id, task_id, request.to_changes())
    return TaskResponse.of(task)


@tasks_router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def delete_task(
    task_id: UUID,
    user: AuthorizedUser = Depends(get_authorized_user),
    use_case: DeleteTaskUseCase = Depends(Provide["tasks.delete_task_use_case"]),
) -> None:
    await use_case.execute_by_id(user.user_id, task_id)


router = APIRouter()
router.include_router(projects_router)
router.include_router(tasks_router)
