from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import date, datetime, tzinfo
from enum import Enum
from typing import Optional, Union
from uuid import UUID

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.arguments import due_instant, is_clear_word, read_choice, read_day
from src.tasks.application.clock import UTC, Clock, utc_now
from src.tasks.application.outcome import (
    INVALID_TASK,
    NO_CHANGES_REQUESTED,
    Outcome,
    TaskPresenter,
    diff_states,
    filter_error,
    ok,
)
from src.tasks.application.task_lookup import TaskLookup
from src.tasks.domain.entities.task import Task
from src.tasks.domain.exceptions import InvalidTaskError
from src.tasks.domain.value_objects.local_calendar import LocalCalendar
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus
from src.tasks.ports.repository_manager import TasksRepositoryManager


class Unset(Enum):
    """Marks a field the caller did not mention, as opposed to one set to ``None``."""

    UNSET = "unset"


UNSET = Unset.UNSET


@dataclass(frozen=True)
class TaskChanges:
    """Typed edits of the id-addressed face. ``UNSET`` leaves a field alone.

    ``None`` is a value: it removes the due date, unfiles the task, or clears the
    notes. A bare day in ``due_at`` means the local midnight that starts it.
    """

    title: Union[str, Unset] = UNSET
    status: Union[TaskStatus, Unset] = UNSET
    priority: Union[TaskPriority, Unset] = UNSET
    due_at: Union[date, datetime, None, Unset] = UNSET
    project_id: Union[UUID, None, Unset] = UNSET
    notes: Union[str, None, Unset] = UNSET


@dataclass(frozen=True)
class TaskChangeRequest:
    """Edits as the text-addressed face receives them. ``None`` leaves a field alone.

    ``set_due_at`` and ``set_project`` accept the word ``none`` to remove the due
    date or to unfile the task; blank ``set_notes`` clears the notes.
    """

    set_status: Optional[str] = None
    set_title: Optional[str] = None
    set_due_at: Optional[str] = None
    set_priority: Optional[str] = None
    set_project: Optional[str] = None
    set_notes: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return all(getattr(self, field.name) is None for field in fields(self))


class UpdateTaskUseCase:
    """Edits one task of the owner.

    ``execute_by_id`` addresses the task by id and answers with the task or a
    domain error. ``execute_by_reference`` addresses it by human text and answers
    with an outcome; it changes nothing unless the text identifies exactly one
    task, and it reports the fields that actually changed, so an edit that
    changed nothing is visible as an empty list. Both apply edits through
    ``_apply``, and neither writes when nothing changed.
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

    async def execute_by_id(self, owner: UserId, task_id: UUID, changes: TaskChanges) -> Task:
        task = await self._lookup.owned_task(owner, task_id)
        if isinstance(changes.project_id, UUID):
            await self._lookup.owned_project(owner, changes.project_id)
        if self._apply(task, changes):
            await self._repositories.tasks.save(task)
        return task

    async def execute_by_reference(
        self,
        owner: UserId,
        reference: str,
        project_scope: Optional[str],
        changes: TaskChangeRequest,
    ) -> Outcome:
        if changes.is_empty:
            return filter_error(
                NO_CHANGES_REQUESTED,
                "Nothing to change: pass at least one of set_status, set_title, set_due_at, "
                "set_priority, set_project, set_notes.",
            )
        typed_changes = self._read_simple_changes(changes)
        if isinstance(typed_changes, Outcome):
            return typed_changes

        projects = await self._repositories.projects.list_by_user(owner)
        presenter = TaskPresenter.for_projects(self._calendar, projects)

        task = await self._lookup.referenced_task(
            owner, reference, project_scope, projects, presenter
        )
        if isinstance(task, Outcome):
            return task

        if changes.set_project is not None:
            if is_clear_word(changes.set_project):
                typed_changes = replace(typed_changes, project_id=None)
            else:
                target = TaskLookup.referenced_project(projects, changes.set_project)
                if isinstance(target, Outcome):
                    return target
                typed_changes = replace(typed_changes, project_id=target.project_id)

        if isinstance(typed_changes.due_at, date) and task.due_at is not None:
            # Text-addressed edits work in whole days: a task already due on
            # that day is left as it is, time of day included.
            if self._calendar.day_of(task.due_at) == typed_changes.due_at:
                typed_changes = replace(typed_changes, due_at=UNSET)

        before = presenter.editable_state(task)
        try:
            changed = self._apply(task, typed_changes)
        except InvalidTaskError as error:
            return filter_error(INVALID_TASK, f"The task was not changed: {error}.")
        if changed:
            await self._repositories.tasks.save(task)

        return ok(
            {
                "updated": presenter.summary(task),
                "changed": diff_states(before, presenter.editable_state(task)),
            }
        )

    @staticmethod
    def _read_simple_changes(changes: TaskChangeRequest) -> Union[TaskChanges, Outcome]:
        """Type the edits that need no lookup; the first bad value stops the update.

        The project is left out: it is a reference, resolved once the task is known.
        """
        status: Union[TaskStatus, Unset] = UNSET
        if changes.set_status is not None:
            chosen_status = read_choice(TaskStatus, changes.set_status, "set_status")
            if isinstance(chosen_status, Outcome):
                return chosen_status
            status = chosen_status

        priority: Union[TaskPriority, Unset] = UNSET
        if changes.set_priority is not None:
            chosen_priority = read_choice(TaskPriority, changes.set_priority, "set_priority")
            if isinstance(chosen_priority, Outcome):
                return chosen_priority
            priority = chosen_priority

        due_at: Union[date, None, Unset] = UNSET
        if changes.set_due_at is not None:
            if is_clear_word(changes.set_due_at):
                due_at = None
            else:
                due_day = read_day(changes.set_due_at, "set_due_at")
                if isinstance(due_day, Outcome):
                    return due_day
                due_at = due_day

        return TaskChanges(
            title=UNSET if changes.set_title is None else changes.set_title,
            status=status,
            priority=priority,
            due_at=due_at,
            notes=UNSET if changes.set_notes is None else changes.set_notes,
        )

    def _apply(self, task: Task, changes: TaskChanges) -> bool:
        """Apply the edits to the entity; ``True`` when its stored state changed."""
        now = self._clock()
        before = _stored_state(task)

        if changes.title is not UNSET:
            task.rename(changes.title)
        if changes.notes is not UNSET:
            task.rewrite_notes(changes.notes)
        if changes.priority is not UNSET:
            task.reprioritise(changes.priority)
        if changes.due_at is not UNSET:
            task.reschedule(
                None if changes.due_at is None else due_instant(self._calendar, changes.due_at)
            )
        if changes.project_id is not UNSET:
            task.move_to_project(changes.project_id)
        if changes.status is not UNSET:
            task.change_status(changes.status, now)

        if _stored_state(task) == before:
            return False
        task.touch(now)
        return True


def _stored_state(task: Task) -> tuple:
    return (
        task.title,
        task.notes,
        task.status,
        task.priority,
        task.due_at,
        task.completed_at,
        task.project_id,
    )
