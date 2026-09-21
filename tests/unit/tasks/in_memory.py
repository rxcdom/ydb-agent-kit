"""In-memory doubles of the tasks ports.

They implement the contract of the ports, not the behaviour of any datastore.
Rows are copied on the way in and on the way out, so an entity a use case
mutates but never saves cannot leak into the stored state. Every write is
logged, which lets a test assert that nothing was written at all.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from uuid import UUID

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.domain.entities.project import Project
from src.tasks.domain.entities.task import Task
from src.tasks.domain.value_objects.date_axis import DateAxis
from src.tasks.ports.project_repository import ProjectRepository
from src.tasks.ports.repository_manager import TasksRepositoryManager, TransactionalOperation
from src.tasks.ports.task_repository import ProjectActivity, TaskRepository, TaskSearchCriteria


class InMemoryProjectRepository(ProjectRepository):
    def __init__(self, write_log: List[Tuple[str, str]]):
        self._rows: Dict[UUID, Project] = {}
        self._write_log = write_log

    def snapshot(self) -> Dict[UUID, Project]:
        """A deep copy of everything stored, for before/after comparisons."""
        return copy.deepcopy(self._rows)

    async def save(self, project: Project, tx: Any = None) -> None:
        self._write_log.append(("save_project", project.name))
        self._rows[project.project_id] = copy.deepcopy(project)

    async def find_by_id(self, user_id: UserId, project_id: UUID) -> Optional[Project]:
        project = self._rows.get(project_id)
        if project is None or project.user_id != user_id:
            return None
        return copy.deepcopy(project)

    async def list_by_user(self, user_id: UserId) -> List[Project]:
        owned = [project for project in self._rows.values() if project.user_id == user_id]
        owned.sort(key=lambda project: (project.name, str(project.project_id)))
        return copy.deepcopy(owned)


class InMemoryTaskRepository(TaskRepository):
    def __init__(self, write_log: List[Tuple[str, str]]):
        self._rows: Dict[UUID, Task] = {}
        self._write_log = write_log

    def snapshot(self) -> Dict[UUID, Task]:
        """A deep copy of everything stored, for before/after comparisons."""
        return copy.deepcopy(self._rows)

    async def save(self, task: Task, tx: Any = None) -> None:
        task.check_invariants()
        self._write_log.append(("save_task", task.title))
        self._rows[task.task_id] = copy.deepcopy(task)

    async def find_by_id(self, user_id: UserId, task_id: UUID) -> Optional[Task]:
        task = self._rows.get(task_id)
        if task is None or task.user_id != user_id:
            return None
        return copy.deepcopy(task)

    async def delete(self, user_id: UserId, task_id: UUID) -> None:
        task = self._rows.get(task_id)
        if task is not None and task.user_id == user_id:
            self._write_log.append(("delete_task", task.title))
            del self._rows[task_id]

    async def list_by_user(self, user_id: UserId) -> List[Task]:
        return copy.deepcopy(self._owned(user_id))

    async def list_by_project(self, user_id: UserId, project_id: UUID) -> List[Task]:
        return copy.deepcopy(
            [task for task in self._owned(user_id) if task.project_id == project_id]
        )

    async def search(
        self,
        user_id: UserId,
        criteria: TaskSearchCriteria,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
        descending: bool = False,
    ) -> List[Task]:
        rows = self._matching(user_id, criteria, with_axis_date=True)
        rows.sort(
            key=lambda task: (task.date_on(criteria.axis), str(task.task_id)), reverse=descending
        )
        page = rows[offset:] if limit is None else rows[offset : offset + limit]
        return copy.deepcopy(page)

    async def count(self, user_id: UserId, criteria: TaskSearchCriteria) -> int:
        return len(self._matching(user_id, criteria, with_axis_date=True))

    async def count_without_axis_date(self, user_id: UserId, criteria: TaskSearchCriteria) -> int:
        return len(self._matching(user_id, criteria, with_axis_date=False))

    async def date_bounds(self, user_id: UserId, axis: DateAxis):
        instants = [
            task.date_on(axis) for task in self._owned(user_id) if task.date_on(axis) is not None
        ]
        return (min(instants), max(instants)) if instants else None

    async def summarise_by_project(self, user_id: UserId) -> List[ProjectActivity]:
        groups: Dict[Tuple[Optional[UUID], Any], List[Task]] = {}
        for task in self._owned(user_id):
            groups.setdefault((task.project_id, task.status), []).append(task)
        return [
            ProjectActivity(
                project_id=project_id,
                status=status,
                task_count=len(tasks),
                first_created_at=min(task.created_at for task in tasks),
                last_updated_at=max(task.updated_at for task in tasks),
            )
            for (project_id, status), tasks in groups.items()
        ]

    def _owned(self, user_id: UserId) -> List[Task]:
        owned = [task for task in self._rows.values() if task.user_id == user_id]
        owned.sort(key=lambda task: (task.created_at, str(task.task_id)))
        return owned

    def _matching(
        self, user_id: UserId, criteria: TaskSearchCriteria, *, with_axis_date: bool
    ) -> List[Task]:
        def matches(task: Task) -> bool:
            instant = task.date_on(criteria.axis)
            if with_axis_date:
                if instant is None:
                    return False
                if criteria.window_start is not None and instant < criteria.window_start:
                    return False
                if criteria.window_end is not None and instant >= criteria.window_end:
                    return False
            elif instant is not None:
                return False
            if criteria.project_id is not None and task.project_id != criteria.project_id:
                return False
            if criteria.statuses is not None and task.status not in criteria.statuses:
                return False
            if criteria.priorities is not None and task.priority not in criteria.priorities:
                return False
            if criteria.text is not None:
                needle = criteria.text.lower()
                if needle not in task.title.lower() and needle not in (task.notes or "").lower():
                    return False
            return True

        return [task for task in self._owned(user_id) if matches(task)]


class InMemoryTasksRepositoryManager(TasksRepositoryManager):
    def __init__(self) -> None:
        self.write_log: List[Tuple[str, str]] = []
        self.transactions = 0
        self._projects = InMemoryProjectRepository(self.write_log)
        self._tasks = InMemoryTaskRepository(self.write_log)

    @property
    def projects(self) -> InMemoryProjectRepository:
        return self._projects

    @property
    def tasks(self) -> InMemoryTaskRepository:
        return self._tasks

    async def given(self, *rows: Union[Project, Task]) -> None:
        """Store fixture rows and leave the write log empty, as if they had always been there."""
        for row in rows:
            if isinstance(row, Project):
                await self._projects.save(row)
            else:
                await self._tasks.save(row)
        self.write_log.clear()

    def snapshot(self) -> Tuple[Dict[UUID, Project], Dict[UUID, Task]]:
        """Every stored project and task of every owner, for before/after comparisons."""
        return self._projects.snapshot(), self._tasks.snapshot()

    def stored_ids(self) -> List[str]:
        """Every row id and owner id in the store; none of them may reach an envelope."""
        projects, tasks = self.snapshot()
        rows = list(projects.values()) + list(tasks.values())
        return sorted(
            {str(project_id) for project_id in projects}
            | {str(task_id) for task_id in tasks}
            | {str(row.user_id) for row in rows}
        )

    async def execute_in_transaction(
        self, operations: Sequence[TransactionalOperation]
    ) -> List[Any]:
        self.transactions += 1
        transaction_handle = object()
        return [await operation(transaction_handle) for operation in operations]
