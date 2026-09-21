from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, timedelta, tzinfo
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union
from uuid import UUID

from src.shared.domain.exceptions import ValidationError
from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.arguments import read_choice, read_choices
from src.tasks.application.clock import UTC
from src.tasks.application.outcome import (
    COMPLETED_AXIS_EXCLUDES_STATUSES,
    Outcome,
    TaskPresenter,
    ambiguous_projects,
    coverage_gap,
    empty_filter,
    filter_error,
    no_data,
    no_records,
    tasks_found,
)
from src.tasks.domain.entities.project import Project
from src.tasks.domain.entities.task import Task
from src.tasks.domain.services.date_window import (
    CoverageGap,
    CoverageWindow,
    DateWindow,
    InvalidWindow,
    RequestedWindow,
    Window,
)
from src.tasks.domain.services.project_reference_resolver import ProjectReferenceResolver
from src.tasks.domain.services.reference_resolution import Ambiguous, Resolved
from src.tasks.domain.value_objects.date_axis import DateAxis
from src.tasks.domain.value_objects.local_calendar import LocalCalendar
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus
from src.tasks.ports.repository_manager import TasksRepositoryManager
from src.tasks.ports.task_repository import TaskSearchCriteria

# Rows a list view returns at most; the envelope says when more exist.
LIST_VIEW_CAP = 50

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200


class GroupBy(str, Enum):
    NONE = "none"
    PROJECT = "project"
    STATUS = "status"
    PRIORITY = "priority"
    WEEK = "week"
    MONTH = "month"


class TaskView(str, Enum):
    SUMMARY = "summary"
    LIST = "list"


@dataclass(frozen=True)
class TaskPage:
    """One page of the id-addressed task listing."""

    tasks: List[Task]
    total_count: int
    has_more: bool


@dataclass(frozen=True)
class _Question:
    """The typed form of a text-addressed query."""

    axis: DateAxis
    requested: RequestedWindow
    project: Optional[str]
    statuses: Optional[FrozenSet[TaskStatus]]
    priorities: Optional[FrozenSet[TaskPriority]]
    text: Optional[str]
    group_by: GroupBy
    view: TaskView


@dataclass
class _Group:
    key: Optional[str]
    order: Tuple
    status_counts: Counter = field(default_factory=Counter)


class QueryTasksUseCase:
    """Answers "what exists, what happened and when" about the owner's tasks.

    ``execute`` serves the text-addressed face and answers with an outcome. The
    window and every filter are evaluated by the datastore through the index of
    the chosen axis; the rows that remain are counted and grouped here, because
    a week or a month is a span of local days and the zone is application
    knowledge. ``list_page`` serves the id-addressed API with plain paging.

    Both faces read days the same way: a window ``[date_from, date_to]`` is
    inclusive and runs from the local midnight that starts ``date_from`` to the
    local midnight that ends ``date_to``.
    """

    def __init__(self, repositories: TasksRepositoryManager, timezone: tzinfo = UTC):
        self._repositories = repositories
        self._calendar = LocalCalendar(timezone)

    async def execute(
        self,
        owner: UserId,
        *,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        date_field: str = DateAxis.CREATED.value,
        project: Optional[str] = None,
        statuses: Optional[Sequence[str]] = None,
        priority: Optional[Sequence[str]] = None,
        text: Optional[str] = None,
        group_by: str = GroupBy.NONE.value,
        view: str = TaskView.SUMMARY.value,
    ) -> Outcome:
        """Arguments are checked before any data is read, so a bad call always gets
        ``filter_error``. In a list view ``group_by`` has no effect: every row
        already carries the fields it could be grouped by.
        """
        question = self._read_question(
            date_from, date_to, date_field, project, statuses, priority, text, group_by, view
        )
        if isinstance(question, Outcome):
            return question

        tasks = self._repositories.tasks
        projects = await self._repositories.projects.list_by_user(owner)
        if await tasks.count(owner, TaskSearchCriteria()) == 0:
            return no_data(projects_count=len(projects))

        bounds = await tasks.date_bounds(owner, question.axis)
        coverage = (
            None
            if bounds is None
            else CoverageWindow.for_axis(question.axis, bounds[0], bounds[1], self._calendar)
        )
        window = question.requested.within(coverage)
        if isinstance(window, CoverageGap):
            return coverage_gap(window, question.axis)

        window_start, window_end = self._calendar.span_of(window.date_from, window.date_to)
        criteria = TaskSearchCriteria(
            axis=question.axis,
            window_start=window_start,
            window_end=window_end,
            statuses=question.statuses,
            priorities=question.priorities,
            text=question.text,
        )

        project_resolved: Optional[Project] = None
        if question.project is not None:
            resolution = ProjectReferenceResolver.resolve(projects, question.project)
            if isinstance(resolution, Ambiguous):
                return ambiguous_projects(question.project, resolution.candidates)
            if not isinstance(resolution, Resolved):
                return await self._nothing_matched(
                    owner,
                    criteria,
                    coverage,
                    window,
                    projects,
                    note=(
                        f"No project matches {question.project!r}, so nothing could be "
                        "filtered by it. The user's projects are listed in available_projects."
                    ),
                )
            project_resolved = resolution.match
            criteria = replace(criteria, project_id=project_resolved.project_id)

        total_count = await tasks.count(owner, criteria)
        if total_count == 0:
            if criteria.has_narrowing_filter:
                return await self._nothing_matched(
                    owner,
                    criteria,
                    coverage,
                    window,
                    projects,
                    note=(
                        "No task passes the filters inside this window. count_without_filters "
                        "is how many tasks the window holds once the filters are dropped, and "
                        "available_statuses are the statuses among those. Loosen one filter or "
                        "tell the user that nothing matched."
                    ),
                )
            return no_records(
                coverage=coverage,
                window_used=window,
                count_without_window=await tasks.count(owner, criteria.without_window()),
            )

        presenter = TaskPresenter.for_projects(self._calendar, projects)
        if question.view is TaskView.LIST:
            rows = await tasks.search(owner, criteria, limit=LIST_VIEW_CAP)
            view_payload: Dict[str, Any] = {
                "tasks": [presenter.summary(task) for task in rows],
                "truncated": total_count > len(rows),
            }
        else:
            rows = await tasks.search(owner, criteria)
            view_payload = {"status_counts": _status_counts(Counter(t.status for t in rows))}
            if question.group_by is not GroupBy.NONE:
                view_payload["groups"] = self._groups(rows, question, presenter)

        return tasks_found(
            window_used=window,
            coverage=coverage,
            date_field=question.axis,
            project_resolved=project_resolved,
            total_count=total_count,
            excluded_without_date=await tasks.count_without_axis_date(
                owner, _axis_population(criteria)
            ),
            view_payload=view_payload,
        )

    async def list_page(
        self,
        owner: UserId,
        *,
        status: Optional[TaskStatus] = None,
        project_id: Optional[UUID] = None,
        date_field: DateAxis = DateAxis.CREATED,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
    ) -> TaskPage:
        """Newest first on the chosen axis; tasks without a value on it are not listed.

        Raises ``InvalidDateWindowError`` for an inverted or unsupported window.
        """
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValidationError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        if offset < 0:
            raise ValidationError("offset cannot be negative")

        requested = RequestedWindow(date_from=date_from, date_to=date_to)
        criteria = TaskSearchCriteria(
            axis=date_field,
            window_start=(
                None if requested.date_from is None
                else self._calendar.start_of(requested.date_from)
            ),
            window_end=(
                None if requested.date_to is None else self._calendar.end_of(requested.date_to)
            ),
            project_id=project_id,
            statuses=None if status is None else frozenset({status}),
        )
        tasks = self._repositories.tasks
        total_count = await tasks.count(owner, criteria)
        rows = await tasks.search(owner, criteria, limit=limit, offset=offset, descending=True)
        return TaskPage(
            tasks=rows, total_count=total_count, has_more=offset + len(rows) < total_count
        )

    def _read_question(
        self,
        date_from: Optional[str],
        date_to: Optional[str],
        date_field: str,
        project: Optional[str],
        statuses: Optional[Sequence[str]],
        priority: Optional[Sequence[str]],
        text: Optional[str],
        group_by: str,
        view: str,
    ) -> Union[_Question, Outcome]:
        axis = read_choice(DateAxis, date_field, "date_field")
        if isinstance(axis, Outcome):
            return axis
        grouping = read_choice(GroupBy, group_by, "group_by")
        if isinstance(grouping, Outcome):
            return grouping
        chosen_view = read_choice(TaskView, view, "view")
        if isinstance(chosen_view, Outcome):
            return chosen_view
        chosen_statuses = read_choices(TaskStatus, statuses, "statuses")
        if isinstance(chosen_statuses, Outcome):
            return chosen_statuses
        priorities = read_choices(TaskPriority, priority, "priority")
        if isinstance(priorities, Outcome):
            return priorities

        requested = DateWindow.parse(date_from, date_to)
        if isinstance(requested, InvalidWindow):
            return filter_error(requested.error_code, requested.message)

        narrowing_statuses = _narrowing_statuses(chosen_statuses, axis)
        if isinstance(narrowing_statuses, Outcome):
            return narrowing_statuses

        return _Question(
            axis=axis,
            requested=requested,
            project=_blank_to_none(project),
            statuses=narrowing_statuses,
            priorities=_narrowing_priorities(priorities),
            text=_blank_to_none(text),
            group_by=grouping,
            view=chosen_view,
        )

    async def _nothing_matched(
        self,
        owner: UserId,
        criteria: TaskSearchCriteria,
        coverage: CoverageWindow,
        window: Window,
        projects: Sequence[Project],
        note: str,
    ) -> Outcome:
        unfiltered = await self._repositories.tasks.search(owner, criteria.without_filters())
        present = {task.status for task in unfiltered}
        return empty_filter(
            coverage=coverage,
            window_used=window,
            count_without_filters=len(unfiltered),
            available_projects=projects,
            available_statuses=[status for status in TaskStatus if status in present],
            note=note,
        )

    def _groups(
        self, rows: Iterable[Task], question: _Question, presenter: TaskPresenter
    ) -> List[Dict[str, Any]]:
        """``[{key, count, status_counts}]`` in reading order.

        Projects group by identity and are labelled by name, so two projects that
        share a name stay two groups; unfiled tasks form the group with a null
        key. A week is keyed by the Monday that starts it, a month by ``YYYY-MM``.
        """
        groups: Dict[Any, _Group] = {}
        for task in rows:
            identity, group = self._group_of(task, question, presenter)
            groups.setdefault(identity, group).status_counts[task.status] += 1
        return [
            {
                "key": group.key,
                "count": sum(group.status_counts.values()),
                "status_counts": _status_counts(group.status_counts),
            }
            for group in sorted(groups.values(), key=lambda group: group.order)
        ]

    def _group_of(
        self, task: Task, question: _Question, presenter: TaskPresenter
    ) -> Tuple[Any, _Group]:
        if question.group_by is GroupBy.PROJECT:
            name = presenter.project_name(task)
            order = (name is None, (name or "").casefold(), str(task.project_id))
            return task.project_id, _Group(key=name, order=order)
        if question.group_by is GroupBy.STATUS:
            return task.status, _Group(task.status.value, (list(TaskStatus).index(task.status),))
        if question.group_by is GroupBy.PRIORITY:
            rank = list(TaskPriority).index(task.priority)
            return task.priority, _Group(task.priority.value, (rank,))

        day = self._calendar.day_of(task.date_on(question.axis))
        if question.group_by is GroupBy.WEEK:
            monday = day - timedelta(days=day.weekday())
            return monday, _Group(monday.isoformat(), (monday,))
        month = f"{day.year:04d}-{day.month:02d}"
        return month, _Group(month, (month,))


def _blank_to_none(text: Optional[str]) -> Optional[str]:
    return (text or "").strip() or None


def _status_counts(counts: Counter) -> Dict[str, int]:
    return {status.value: counts.get(status, 0) for status in TaskStatus}


def _narrowing_statuses(
    requested: Optional[FrozenSet[TaskStatus]], axis: DateAxis
) -> Union[Optional[FrozenSet[TaskStatus]], Outcome]:
    """The status filter as far as it can narrow anything on this axis.

    A filter that leaves no status the axis can hold contradicts the axis. One
    that keeps every status the axis can hold narrows nothing and is dropped, so
    an empty result is not blamed on it.
    """
    if requested is None:
        return None
    on_axis = axis.statuses_with_value
    effective = requested & on_axis
    if not effective:
        held = ", ".join(sorted(status.value for status in on_axis))
        asked = ", ".join(sorted(status.value for status in requested))
        return filter_error(
            COMPLETED_AXIS_EXCLUDES_STATUSES,
            f"date_field={axis.value} only holds tasks with status {held}, but statuses asks "
            f"for {asked}. Drop statuses, or pick the date_field the question is about: "
            "created for when tasks were added, due for when they are due.",
        )
    return None if effective == on_axis else effective


def _narrowing_priorities(
    requested: Optional[FrozenSet[TaskPriority]],
) -> Optional[FrozenSet[TaskPriority]]:
    return None if requested is None or requested == frozenset(TaskPriority) else requested


def _axis_population(criteria: TaskSearchCriteria) -> TaskSearchCriteria:
    """The criteria restricted to tasks that could have a value on the axis at all.

    Counting tasks "left out for having no date" only makes sense among those:
    an open task is not missing a completion date, it simply is not finished.
    """
    on_axis = criteria.axis.statuses_with_value
    if criteria.statuses is not None or on_axis == frozenset(TaskStatus):
        return criteria
    return replace(criteria, statuses=on_axis)
