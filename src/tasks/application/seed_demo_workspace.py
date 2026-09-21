"""A synthetic demo workspace: four projects and about sixty tasks.

The data exists to make the interesting questions answerable. Two project names
share a word, three task titles share another, some open tasks are overdue, one
three-week stretch is empty on every date axis, and recent days and the previous
calendar month both hold finished work. Every date is an offset from an anchor
day, so the workspace looks the same whenever it is generated.

The rows every demo scenario relies on are pinned with explicit offsets. The
rest are drawn from a random generator seeded per row, inside ranges that cannot
disturb what the pinned rows guarantee.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Dict, List, Optional, Tuple
from uuid import NAMESPACE_DNS, UUID, uuid5

from src.shared.domain.exceptions import ValidationError
from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.clock import UTC
from src.tasks.domain.entities.project import Project
from src.tasks.domain.entities.task import Task
from src.tasks.domain.value_objects.local_calendar import LocalCalendar, is_supported_day
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus
from src.tasks.ports.repository_manager import TasksRepositoryManager

DEMO_SEED = 20260901

# Tasks are created within this many days before the anchor, never earlier.
CREATED_SPAN_DAYS = 270
# Nothing is created, due or completed from 130 to 110 days before the anchor.
QUIET_STRETCH_FIRST_DAYS_AGO = 130
QUIET_STRETCH_LAST_DAYS_AGO = 110
# Finished work is pinned to every second day of the last two weeks ...
RECENT_COMPLETION_DAYS_AGO = (1, 3, 5, 7, 9, 11, 13)
# ... and to this many days spread over the previous calendar month.
PREVIOUS_MONTH_COMPLETIONS = 6
# The furthest a generated due date reaches past the anchor.
LATEST_DUE_IN_DAYS = 60

HOME_RENOVATION = "Home renovation"
HOME_OFFICE = "Home office"
CLIENT_WORK = "Client work"
GARDEN = "Garden"

PROJECT_DESCRIPTIONS: Dict[str, str] = {
    HOME_RENOVATION: "Kitchen and bathroom remodel, one room at a time.",
    HOME_OFFICE: "Desk setup, equipment and filing for the spare room.",
    CLIENT_WORK: "Deliverables, invoices and meetings for consulting clients.",
    GARDEN: "Seasonal planting, beds and tools.",
}

_ID_NAMESPACE = uuid5(NAMESPACE_DNS, "demo-workspace.ydb-agent-kit")

_LOW, _NORMAL, _HIGH = TaskPriority.LOW, TaskPriority.NORMAL, TaskPriority.HIGH


@dataclass(frozen=True)
class _Blueprint:
    """One task as offsets from the anchor day. ``due_in_days`` below zero is a past due day."""

    title: str
    project: Optional[str]
    status: TaskStatus
    priority: TaskPriority
    created_days_ago: int
    due_in_days: Optional[int] = None
    completed_days_ago: Optional[int] = None
    previous_month_slot: Optional[int] = None
    notes: Optional[str] = None


def _open(
    title: str,
    project: Optional[str],
    priority: TaskPriority,
    created: int,
    due_in: Optional[int] = None,
    notes: Optional[str] = None,
) -> _Blueprint:
    return _Blueprint(title, project, TaskStatus.OPEN, priority, created, due_in, notes=notes)


def _done(
    title: str,
    project: Optional[str],
    priority: TaskPriority,
    created: int,
    completed: Optional[int] = None,
    slot: Optional[int] = None,
    due_in: Optional[int] = None,
) -> _Blueprint:
    return _Blueprint(
        title,
        project,
        TaskStatus.DONE,
        priority,
        created,
        due_in,
        completed_days_ago=completed,
        previous_month_slot=slot,
    )


def _cancelled(
    title: str,
    project: Optional[str],
    priority: TaskPriority,
    created: int,
    due_in: Optional[int] = None,
) -> _Blueprint:
    return _Blueprint(title, project, TaskStatus.CANCELLED, priority, created, due_in)


_PINNED: Tuple[_Blueprint, ...] = (
    # Uniquely titled open tasks the walkthroughs address by name.
    _open("Book dentist appointment", None, _NORMAL, 12),
    _open(
        "Order kitchen tiles", HOME_RENOVATION, _NORMAL, 20, due_in=9,
        notes="Matte white, 20 by 20 cm. Twelve square metres plus one spare box.",
    ),
    _open(
        "Renew domain name", CLIENT_WORK, _HIGH, 34, due_in=21,
        notes="The registrar sends a reminder thirty days before expiry.",
    ),
    # Three open titles sharing one word: a reference to it has to come back ambiguous.
    _open("Quarterly report", CLIENT_WORK, _HIGH, 25, due_in=12),
    _open("Expense report", CLIENT_WORK, _NORMAL, 30, due_in=-4),
    _open("Weekly status report", CLIENT_WORK, _NORMAL, 6, due_in=3),
    # Overdue: open and due before the anchor, in several projects.
    _open("Fix leaking bathroom tap", HOME_RENOVATION, _HIGH, 40, due_in=-9),
    _open("Get quotes for floor sanding", HOME_RENOVATION, _NORMAL, 16, due_in=-2),
    _open("Replace desk lamp", HOME_OFFICE, _LOW, 22, due_in=-6),
    _open("Send invoice for the design workshop", CLIENT_WORK, _HIGH, 15, due_in=-1),
    _open("Prune the apple tree", GARDEN, _NORMAL, 50, due_in=-15),
    # Finished on every second day of the last two weeks.
    _done("Paint hallway ceiling", HOME_RENOVATION, _NORMAL, 18, completed=1, due_in=-2),
    _done("Assemble standing desk", HOME_OFFICE, _NORMAL, 21, completed=3),
    _done("Prepare workshop slides", CLIENT_WORK, _HIGH, 19, completed=5, due_in=-5),
    _done("Plant herb seedlings", GARDEN, _LOW, 15, completed=7),
    _done("Return library books", None, _LOW, 14, completed=9, due_in=-8),
    _done("Seal bathroom grout", HOME_RENOVATION, _NORMAL, 28, completed=11),
    _done("Send project proposal", CLIENT_WORK, _HIGH, 26, completed=13, due_in=-12),
    # Finished on days spread over the previous calendar month.
    _done("Remove old kitchen cabinets", HOME_RENOVATION, _HIGH, 100, slot=0),
    _done("Install shelf brackets", HOME_OFFICE, _NORMAL, 95, slot=1),
    _done("Draft client onboarding checklist", CLIENT_WORK, _NORMAL, 90, slot=2),
    _done("Build raised vegetable bed", GARDEN, _NORMAL, 85, slot=3),
    _done("Compare electricity tariffs", None, _LOW, 80, slot=4),
    _done("Review contract renewal terms", CLIENT_WORK, _HIGH, 75, slot=5),
    # Given up on.
    _cancelled("Hire a skip for rubble", HOME_RENOVATION, _LOW, 62),
    _cancelled("Buy a second monitor", HOME_OFFICE, _NORMAL, 45, due_in=-20),
    _cancelled("Plan conference trip", CLIENT_WORK, _NORMAL, 150),
    # Old history, so the workspace reaches back over the whole span.
    _done("Measure kitchen for the new layout", HOME_RENOVATION, _HIGH, 268, completed=259),
    _done("Register consulting business", CLIENT_WORK, _HIGH, 262, completed=247),
    _done("Clear out the garden shed", GARDEN, _NORMAL, 215, completed=204),
    _done("Set up document scanner", HOME_OFFICE, _NORMAL, 170, completed=161),
)

# Titles whose status, priority and dates are drawn from the seeded generator.
_DRAWN_TITLES: Tuple[Tuple[Optional[str], Tuple[str, ...]], ...] = (
    (
        HOME_RENOVATION,
        (
            "Strip wallpaper in the spare room",
            "Replace kitchen light fittings",
            "Order bathroom mirror",
            "Fit new door handles",
            "Book electrician for rewiring",
            "Sand window frames",
            "Pick worktop material",
        ),
    ),
    (
        HOME_OFFICE,
        (
            "Tidy cable management",
            "Back up laptop",
            "Label filing folders",
            "Replace office chair wheels",
            "Set up printer",
            "Mount whiteboard",
        ),
    ),
    (
        CLIENT_WORK,
        (
            "Schedule kickoff call",
            "Update portfolio page",
            "Send meeting notes",
            "Prepare pricing sheet",
            "Follow up on unpaid invoice",
            "Archive finished project files",
            "Book meeting room",
        ),
    ),
    (
        GARDEN,
        (
            "Mow the lawn",
            "Repair fence panel",
            "Order compost",
            "Clean the gutters",
            "Sharpen garden shears",
        ),
    ),
    (
        None,
        (
            "Renew passport",
            "Book car service",
            "Buy birthday gift for Sam",
            "Donate old clothes",
        ),
    ),
)


@dataclass(frozen=True)
class DemoWorkspace:
    anchor_date: date
    projects: Tuple[Project, ...]
    tasks: Tuple[Task, ...]


@dataclass(frozen=True)
class SeedSummary:
    """What a seeding run wrote, for the command that reports it."""

    anchor_date: date
    projects_count: int
    tasks_count: int
    open_count: int
    done_count: int
    cancelled_count: int
    overdue_count: int


def build_demo_workspace(
    owner: UserId,
    anchor_date: date,
    seed: int = DEMO_SEED,
    *,
    timezone: tzinfo = UTC,
) -> DemoWorkspace:
    """The demo workspace of ``owner`` around ``anchor_date``; a pure function.

    The same owner, anchor, seed and zone always give the same rows. Ids derive
    from the owner and the row's name, so generating twice for one owner yields
    the same ids and different owners never share one. Days are local days of
    ``timezone``; a due date is the local midnight that starts its day.
    """
    earliest = anchor_date - timedelta(days=CREATED_SPAN_DAYS)
    latest = anchor_date + timedelta(days=LATEST_DUE_IN_DAYS)
    if not (is_supported_day(earliest) and is_supported_day(latest)):
        raise ValidationError(
            f"anchor_date {anchor_date.isoformat()} is too close to the edge of the "
            "supported calendar range"
        )

    calendar = LocalCalendar(timezone)
    project_created_at = calendar.start_of(earliest)
    projects = tuple(
        Project(
            project_id=uuid5(_ID_NAMESPACE, f"{owner}/project/{name}"),
            user_id=owner,
            name=name,
            description=description,
            created_at=project_created_at,
            updated_at=project_created_at,
        )
        for name, description in PROJECT_DESCRIPTIONS.items()
    )
    project_ids = {project.name: project.project_id for project in projects}

    blueprints: List[_Blueprint] = list(_PINNED)
    for project_name, titles in _DRAWN_TITLES:
        for title in titles:
            blueprints.append(_draw(title, project_name, random.Random(f"{seed}/draw/{title}")))

    tasks = tuple(
        _materialise(
            blueprint,
            owner,
            anchor_date,
            calendar,
            project_ids,
            random.Random(f"{seed}/clock/{blueprint.title}"),
        )
        for blueprint in blueprints
    )
    return DemoWorkspace(anchor_date=anchor_date, projects=projects, tasks=tasks)


def _is_quiet(days_ago: int) -> bool:
    return QUIET_STRETCH_LAST_DAYS_AGO <= days_ago <= QUIET_STRETCH_FIRST_DAYS_AGO


def _after_quiet_stretch(days_ago: int) -> int:
    """Move a day out of the quiet stretch to the first day after it.

    Moving forward in time keeps every "happened after it was created" relation.
    """
    return QUIET_STRETCH_LAST_DAYS_AGO - 1 if _is_quiet(days_ago) else days_ago


def _active_days_ago(earliest_days_ago: int, latest_days_ago: int) -> List[int]:
    return [
        days_ago
        for days_ago in range(latest_days_ago, earliest_days_ago + 1)
        if not _is_quiet(days_ago)
    ]


def _draw(title: str, project: Optional[str], rng: random.Random) -> _Blueprint:
    """Draw a task inside ranges that keep it clear of the quiet stretch and the span."""
    status = rng.choices(
        (TaskStatus.DONE, TaskStatus.OPEN, TaskStatus.CANCELLED), weights=(6, 3, 1)
    )[0]
    priority = rng.choices((_LOW, _NORMAL, _HIGH), weights=(2, 5, 3))[0]

    if status is TaskStatus.OPEN:
        created = rng.randint(2, 90)
        due_style = rng.choices(("none", "upcoming", "overdue"), weights=(3, 5, 2))[0]
        due_in: Optional[int] = None
        if due_style == "upcoming":
            due_in = rng.randint(1, LATEST_DUE_IN_DAYS)
        elif due_style == "overdue":
            due_in = -rng.randint(1, min(created - 1, 30))
        return _Blueprint(title, project, status, priority, created, due_in)

    if status is TaskStatus.CANCELLED:
        created = rng.choice(_active_days_ago(CREATED_SPAN_DAYS, 20))
        return _Blueprint(title, project, status, priority, created)

    created = rng.choice(_active_days_ago(CREATED_SPAN_DAYS, 14))
    completed = _after_quiet_stretch(max(1, created - rng.randint(1, 21)))
    due_in = None
    if rng.random() < 0.4:
        due_days_ago = min(completed + rng.randint(-7, 7), created - 1)
        due_in = -_after_quiet_stretch(due_days_ago)
    return _Blueprint(title, project, status, priority, created, due_in, completed)


def _previous_month_day(anchor_date: date, slot: int) -> date:
    """Day ``slot`` of the days spread evenly over the calendar month before the anchor's."""
    last_day = anchor_date.replace(day=1) - timedelta(days=1)
    first_day = last_day.replace(day=1)
    step = (last_day - first_day).days * slot // (PREVIOUS_MONTH_COMPLETIONS - 1)
    return first_day + timedelta(days=step)


def _materialise(
    blueprint: _Blueprint,
    owner: UserId,
    anchor_date: date,
    calendar: LocalCalendar,
    project_ids: Dict[str, UUID],
    rng: random.Random,
) -> Task:
    def moment(day: date, first_hour: int, last_hour: int) -> datetime:
        wall_clock = time(rng.randint(first_hour, last_hour), rng.randrange(60))
        return calendar.instant_of(datetime.combine(day, wall_clock))

    created_at = moment(anchor_date - timedelta(days=blueprint.created_days_ago), 8, 12)

    completed_at: Optional[datetime] = None
    if blueprint.previous_month_slot is not None:
        completed_at = moment(
            _previous_month_day(anchor_date, blueprint.previous_month_slot), 13, 19
        )
    elif blueprint.completed_days_ago is not None:
        completed_at = moment(anchor_date - timedelta(days=blueprint.completed_days_ago), 13, 19)

    due_at = (
        None
        if blueprint.due_in_days is None
        else calendar.start_of(anchor_date + timedelta(days=blueprint.due_in_days))
    )
    return Task(
        task_id=uuid5(_ID_NAMESPACE, f"{owner}/task/{blueprint.title}"),
        user_id=owner,
        project_id=None if blueprint.project is None else project_ids[blueprint.project],
        title=blueprint.title,
        notes=blueprint.notes,
        status=blueprint.status,
        priority=blueprint.priority,
        due_at=due_at,
        completed_at=completed_at,
        created_at=created_at,
        updated_at=completed_at or created_at,
    )


class SeedDemoWorkspaceUseCase:
    """Writes the demo workspace of one owner.

    All rows go through one transaction, so a failed run leaves no half-seeded
    workspace. Rows are upserted under stable ids: seeding the same owner again
    refreshes the demo rows and touches nothing else the owner has.
    """

    def __init__(self, repositories: TasksRepositoryManager, timezone: tzinfo = UTC):
        self._repositories = repositories
        self._timezone = timezone

    async def execute(self, owner: UserId, anchor_date: date) -> SeedSummary:
        workspace = build_demo_workspace(owner, anchor_date, timezone=self._timezone)

        def save_project(project: Project):
            return lambda tx: self._repositories.projects.save(project, tx)

        def save_task(task: Task):
            return lambda tx: self._repositories.tasks.save(task, tx)

        await self._repositories.execute_in_transaction(
            [save_project(project) for project in workspace.projects]
            + [save_task(task) for task in workspace.tasks]
        )
        return summarise(workspace, LocalCalendar(self._timezone))


def summarise(workspace: DemoWorkspace, calendar: LocalCalendar) -> SeedSummary:
    anchor_start = calendar.start_of(workspace.anchor_date)

    def count(status: TaskStatus) -> int:
        return sum(1 for task in workspace.tasks if task.status is status)

    return SeedSummary(
        anchor_date=workspace.anchor_date,
        projects_count=len(workspace.projects),
        tasks_count=len(workspace.tasks),
        open_count=count(TaskStatus.OPEN),
        done_count=count(TaskStatus.DONE),
        cancelled_count=count(TaskStatus.CANCELLED),
        overdue_count=sum(
            1
            for task in workspace.tasks
            if task.status is TaskStatus.OPEN
            and task.due_at is not None
            and task.due_at < anchor_start
        ),
    )
