"""The outcome vocabulary of the text-addressed face of the tasks module.

A caller that addresses rows by human text (the agent) never receives an
exception for something the user could have caused. It receives an ``Outcome``:
one of eight status words plus the facts needed to act on it. Zero rows is not
one answer but four (no data at all, a window outside the data, a filter that
matched nothing, an honestly empty window), and the difference decides what can
truthfully be said to the user.

Envelopes are plain JSON values. They carry names and local ``YYYY-MM-DD``
days, never row ids: a caller that cannot name an id cannot be tricked into
using one.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence
from uuid import UUID

from src.tasks.domain.entities.project import Project
from src.tasks.domain.entities.task import Task
from src.tasks.domain.services.date_window import CoverageGap, CoverageWindow, Window
from src.tasks.domain.value_objects.date_axis import DateAxis
from src.tasks.domain.value_objects.local_calendar import LocalCalendar
from src.tasks.domain.value_objects.task_status import TaskStatus

UNKNOWN_VALUE = "unknown_value"
COMPLETED_AXIS_EXCLUDES_STATUSES = "completed_axis_excludes_statuses"
NO_CHANGES_REQUESTED = "no_changes_requested"
INVALID_TASK = "invalid_task"

REFERENCE_KIND_PROJECT = "project"
REFERENCE_KIND_TASK = "task"


class OutcomeStatus(str, Enum):
    OK = "ok"
    NO_DATA = "no_data"
    COVERAGE_GAP = "coverage_gap"
    EMPTY_FILTER = "empty_filter"
    NO_RECORDS = "no_records"
    AMBIGUOUS_SOURCE = "ambiguous_source"
    NOT_FOUND = "not_found"
    FILTER_ERROR = "filter_error"


@dataclass(frozen=True)
class Outcome:
    status: OutcomeStatus
    data: Dict[str, Any]

    def to_envelope(self) -> Dict[str, Any]:
        """The JSON-ready form: ``{"status": "<word>", "data": {...}}``."""
        return {"status": self.status.value, "data": copy.deepcopy(self.data)}


@dataclass(frozen=True)
class TaskPresenter:
    """Renders tasks for envelopes: project names instead of ids, local days instead of instants."""

    calendar: LocalCalendar
    project_names: Mapping[UUID, str]

    @classmethod
    def for_projects(cls, calendar: LocalCalendar, projects: Sequence[Project]) -> "TaskPresenter":
        return cls(calendar, {project.project_id: project.name for project in projects})

    def day(self, instant: Optional[datetime]) -> Optional[str]:
        return None if instant is None else self.calendar.day_of(instant).isoformat()

    def project_name(self, task: Task) -> Optional[str]:
        return None if task.project_id is None else self.project_names.get(task.project_id)

    def summary(self, task: Task) -> Dict[str, Any]:
        return {
            "title": task.title,
            "project": self.project_name(task),
            "status": task.status.value,
            "priority": task.priority.value,
            "due_at": self.day(task.due_at),
            "completed_at": self.day(task.completed_at),
            "created_at": self.day(task.created_at),
        }

    def editable_state(self, task: Task) -> Dict[str, Any]:
        """Every field a write can change, as the caller sees it. Diffed to report changes."""
        return {
            "title": task.title,
            "project": self.project_name(task),
            "status": task.status.value,
            "priority": task.priority.value,
            "due_at": self.day(task.due_at),
            "completed_at": self.day(task.completed_at),
            "notes": task.notes,
        }

    def candidate(self, task: Task) -> Dict[str, Any]:
        return {
            "title_or_name": task.title,
            "project": self.project_name(task),
            "status": task.status.value,
            "due_at": self.day(task.due_at),
        }


def diff_states(before: Mapping[str, Any], after: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """``[{field, from, to}]`` for every field whose rendered value differs."""
    return [
        {"field": field, "from": before[field], "to": after[field]}
        for field in before
        if before[field] != after[field]
    ]


def render_days(date_from: Optional[date], date_to: Optional[date]) -> Dict[str, Optional[str]]:
    return {
        "from": None if date_from is None else date_from.isoformat(),
        "to": None if date_to is None else date_to.isoformat(),
    }


def render_window(window: Window) -> Dict[str, Optional[str]]:
    return render_days(window.date_from, window.date_to)


def render_coverage(coverage: Optional[CoverageWindow]) -> Optional[Dict[str, Optional[str]]]:
    return None if coverage is None else render_days(coverage.first, coverage.last)


def ok(data: Dict[str, Any]) -> Outcome:
    return Outcome(OutcomeStatus.OK, data)


def tasks_found(
    *,
    window_used: Window,
    coverage: CoverageWindow,
    date_field: DateAxis,
    project_resolved: Optional[Project],
    total_count: int,
    excluded_without_date: int,
    view_payload: Dict[str, Any],
) -> Outcome:
    """Rows were found. ``view_payload`` carries ``tasks`` or the summary counts and ``groups``."""
    data: Dict[str, Any] = {
        "window_used": render_window(window_used),
        "coverage": render_coverage(coverage),
        "date_field": date_field.value,
    }
    if project_resolved is not None:
        data["project_resolved"] = {"name": project_resolved.name}
    data["total_count"] = total_count
    data["excluded_without_date"] = excluded_without_date
    data.update(view_payload)
    return ok(data)


def no_data(projects_count: int) -> Outcome:
    """The owner has no tasks at all, whatever the question was."""
    return Outcome(OutcomeStatus.NO_DATA, {"projects_count": projects_count})


def coverage_gap(gap: CoverageGap, date_field: DateAxis) -> Outcome:
    """The requested window cannot hold anything on this axis."""
    if gap.coverage is None:
        note = (
            f"No task has a value on the '{date_field.value}' axis, so no window on it can "
            "contain anything."
        )
    else:
        note = (
            f"The requested window lies entirely outside the days on which the "
            f"'{date_field.value}' axis has data. That data runs from "
            f"{gap.coverage.first.isoformat()} to {gap.coverage.last.isoformat()}; give the user "
            "both of these days."
        )
    return Outcome(
        OutcomeStatus.COVERAGE_GAP,
        {
            "coverage": render_coverage(gap.coverage),
            "requested_window": render_days(gap.requested.date_from, gap.requested.date_to),
            "date_field": date_field.value,
            "note": note,
        },
    )


def empty_filter(
    *,
    coverage: CoverageWindow,
    window_used: Window,
    count_without_filters: int,
    available_projects: Sequence[Project],
    available_statuses: Sequence[TaskStatus],
    note: str,
) -> Outcome:
    """A narrowing filter matched nothing inside a window that overlaps the coverage.

    ``available_projects`` is every project name the owner has, the vocabulary a
    project reference is resolved against. ``available_statuses`` are the
    statuses present in the window once the filters are dropped.
    """
    return Outcome(
        OutcomeStatus.EMPTY_FILTER,
        {
            "coverage": render_coverage(coverage),
            "window_used": render_window(window_used),
            "count_without_filters": count_without_filters,
            "available_projects": [project.name for project in available_projects],
            "available_statuses": [status.value for status in available_statuses],
            "note": note,
        },
    )


def no_records(
    *, coverage: CoverageWindow, window_used: Window, count_without_window: int
) -> Outcome:
    """Nothing happened in the window and no filter is to blame: an honest gap."""
    return Outcome(
        OutcomeStatus.NO_RECORDS,
        {
            "coverage": render_coverage(coverage),
            "window_used": render_window(window_used),
            "count_without_window": count_without_window,
        },
    )


def ambiguous_projects(reference: str, candidates: Sequence[Project]) -> Outcome:
    return Outcome(
        OutcomeStatus.AMBIGUOUS_SOURCE,
        {
            "reference": reference,
            "reference_kind": REFERENCE_KIND_PROJECT,
            "matched": [
                {"title_or_name": project.name, "description": project.description}
                for project in candidates
            ],
            "note": (
                f"The project reference {reference!r} fits {len(candidates)} projects. "
                "Nothing was read or changed for any of them. Ask the user which one they "
                "mean and repeat the call with the exact project name."
            ),
        },
    )


def ambiguous_tasks(
    reference: str, candidates: Sequence[Task], presenter: TaskPresenter
) -> Outcome:
    return Outcome(
        OutcomeStatus.AMBIGUOUS_SOURCE,
        {
            "reference": reference,
            "reference_kind": REFERENCE_KIND_TASK,
            "matched": [presenter.candidate(task) for task in candidates],
            "note": (
                f"The task reference {reference!r} fits {len(candidates)} tasks. Nothing was "
                "changed. Ask the user which one they mean, then repeat the call with the "
                "exact title or narrow it with project_scope."
            ),
        },
    )


def project_not_found(reference: str, available_projects: Sequence[Project]) -> Outcome:
    return Outcome(
        OutcomeStatus.NOT_FOUND,
        {
            "reference": reference,
            "reference_kind": REFERENCE_KIND_PROJECT,
            "available_projects": [project.name for project in available_projects],
            "note": (
                f"No project matches {reference!r}. Nothing was changed. The user's projects "
                "are listed in available_projects; a project cannot be created from here."
            ),
        },
    )


def task_not_found(reference: str, project_scope: Optional[Project]) -> Outcome:
    scope = "" if project_scope is None else f" in the project {project_scope.name!r}"
    return Outcome(
        OutcomeStatus.NOT_FOUND,
        {
            "reference": reference,
            "reference_kind": REFERENCE_KIND_TASK,
            "note": (
                f"No task matches {reference!r}{scope}. Nothing was changed. Look the task up "
                "first or ask the user for its exact title."
            ),
        },
    )


def filter_error(error_code: str, message: str) -> Outcome:
    """The arguments are invalid or contradict each other; ``message`` says how to repair them."""
    return Outcome(OutcomeStatus.FILTER_ERROR, {"error_code": error_code, "message": message})
