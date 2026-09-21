"""The demo workspace: deterministic, and built so that every walkthrough question has an answer.

The properties below are the contract of the generator. They are checked for
several anchor days (a Monday, a Sunday, the first of a month, the first of a
year, a leap day, two month ends) in several zones, because every one of them
is stated in local calendar days relative to the anchor.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest

from src.shared.domain.exceptions import ValidationError
from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.seed_demo_workspace import (
    DEMO_SEED,
    DemoWorkspace,
    SeedDemoWorkspaceUseCase,
    SeedSummary,
    build_demo_workspace,
)
from src.tasks.domain.entities.task import Task
from src.tasks.domain.services.project_reference_resolver import ProjectReferenceResolver
from src.tasks.domain.services.reference_resolution import Ambiguous, Resolved
from src.tasks.domain.services.task_reference_resolver import TaskReferenceResolver
from src.tasks.domain.value_objects.local_calendar import LATEST_SUPPORTED_DAY, LocalCalendar
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus
from tests.unit.tasks.builders import a_project, a_task
from tests.unit.tasks.in_memory import InMemoryTasksRepositoryManager

UTC = timezone.utc
PROJECT_ROOT = Path(__file__).resolve().parents[3]

OWNER = UserId.from_string("0d9c1b9e-6f0a-4a57-9a53-0c2f6f1f7a11")
OTHER_OWNER = UserId.from_string("5b7e0c43-8f7d-4f0e-b0f4-0f6a3f1c2d22")

MONDAY = date(2026, 9, 21)
ANCHORS = [
    MONDAY,
    date(2026, 9, 20),
    date(2026, 3, 1),
    date(2027, 1, 1),
    date(2028, 2, 29),
    date(2026, 3, 31),
    date(2026, 12, 31),
]
ZONES = [UTC, ZoneInfo("Europe/Berlin"), ZoneInfo("Pacific/Auckland")]

PROJECT_NAMES = ["Home renovation", "Home office", "Client work", "Garden"]
REPORT_TITLES = ["Expense report", "Quarterly report", "Weekly status report"]


@dataclass
class Seeded:
    anchor: date
    calendar: LocalCalendar
    workspace: DemoWorkspace

    @property
    def tasks(self) -> List[Task]:
        return list(self.workspace.tasks)

    @property
    def project_names(self) -> Dict[object, str]:
        return {project.project_id: project.name for project in self.workspace.projects}

    def day(self, instant: datetime) -> date:
        return self.calendar.day_of(instant)

    def days_ago(self, days: int) -> date:
        return self.anchor - timedelta(days=days)

    def project_of(self, task: Task) -> Optional[str]:
        return None if task.project_id is None else self.project_names[task.project_id]

    def titled(self, title: str) -> Task:
        (task,) = [task for task in self.tasks if task.title == title]
        return task

    def is_overdue(self, task: Task) -> bool:
        return (
            task.status is TaskStatus.OPEN
            and task.due_at is not None
            and self.day(task.due_at) < self.anchor
        )


def _label(parameters) -> str:
    anchor, zone = parameters
    return f"{anchor.isoformat()}-{getattr(zone, 'key', 'UTC')}"


@pytest.fixture(params=[(anchor, zone) for anchor in ANCHORS for zone in ZONES], ids=_label)
def seeded(request) -> Seeded:
    anchor, zone = request.param
    return Seeded(anchor, LocalCalendar(zone), build_demo_workspace(OWNER, anchor, timezone=zone))


class TestGuaranteedProperties:
    def test_a_exactly_four_named_projects_each_with_a_description(self, seeded):
        projects = seeded.workspace.projects

        assert sorted(project.name for project in projects) == sorted(PROJECT_NAMES)
        assert all(project.description for project in projects)
        assert all(project.user_id == OWNER for project in projects)
        assert len({project.project_id for project in projects}) == 4

    def test_b_about_sixty_tasks_created_within_the_last_270_days(self, seeded):
        assert 55 <= len(seeded.tasks) <= 65
        created_days = [seeded.day(task.created_at) for task in seeded.tasks]
        assert min(created_days) >= seeded.days_ago(270), "nothing is older than the span"
        assert max(created_days) <= seeded.anchor, "nothing is created in the future"
        assert min(created_days) <= seeded.days_ago(240), "the history reaches back nine months"
        assert all(task.user_id == OWNER for task in seeded.tasks)

    def test_b_a_few_tasks_are_unfiled_and_the_rest_sit_in_the_four_projects(self, seeded):
        filed = Counter(seeded.project_of(task) for task in seeded.tasks)

        assert 3 <= filed[None] <= 10
        assert set(filed) == set(PROJECT_NAMES) | {None}
        assert all(filed[name] >= 5 for name in PROJECT_NAMES)

    def test_b_a_mix_of_done_open_and_a_few_cancelled_in_all_three_priorities(self, seeded):
        statuses = Counter(task.status for task in seeded.tasks)
        priorities = {task.priority for task in seeded.tasks}

        assert statuses[TaskStatus.DONE] >= 15
        assert statuses[TaskStatus.OPEN] >= 15
        assert 2 <= statuses[TaskStatus.CANCELLED] <= 8
        assert priorities == set(TaskPriority)

    def test_b_finished_tasks_were_completed_between_their_creation_and_the_anchor(self, seeded):
        for task in seeded.tasks:
            if task.status is TaskStatus.DONE:
                assert task.created_at <= task.completed_at, task.title
                assert seeded.day(task.completed_at) <= seeded.anchor, task.title
            else:
                assert task.completed_at is None, task.title

    def test_b_open_tasks_come_with_a_future_due_day_a_past_one_and_none(self, seeded):
        open_tasks = [task for task in seeded.tasks if task.status is TaskStatus.OPEN]
        due_days = [seeded.day(task.due_at) for task in open_tasks if task.due_at is not None]

        assert any(day > seeded.anchor for day in due_days)
        assert any(day < seeded.anchor for day in due_days)
        assert sum(1 for task in open_tasks if task.due_at is None) >= 2

    def test_b_a_due_date_is_the_local_midnight_that_starts_its_day(self, seeded):
        for task in seeded.tasks:
            if task.due_at is not None:
                due_day = seeded.day(task.due_at)
                assert task.due_at == seeded.calendar.start_of(due_day), task.title
                assert task.due_at >= task.created_at, task.title

    def test_c_at_least_five_overdue_tasks_spread_over_three_projects(self, seeded):
        overdue = [task for task in seeded.tasks if seeded.is_overdue(task)]
        by_project = Counter(seeded.project_of(task) for task in overdue)

        assert len(overdue) >= 5
        for name in ("Home renovation", "Home office", "Client work"):
            assert by_project[name] >= 1, f"no overdue task in {name}"

    def test_d_nothing_happens_on_any_axis_during_the_quiet_stretch(self, seeded):
        first_quiet_day, last_quiet_day = seeded.days_ago(130), seeded.days_ago(110)

        for task in seeded.tasks:
            for axis, instant in (
                ("created_at", task.created_at),
                ("due_at", task.due_at),
                ("completed_at", task.completed_at),
            ):
                if instant is not None:
                    day = seeded.day(instant)
                    assert not first_quiet_day <= day <= last_quiet_day, (task.title, axis, day)

    def test_d_there_is_activity_on_both_sides_of_the_quiet_stretch(self, seeded):
        created_days = [seeded.day(task.created_at) for task in seeded.tasks]

        assert any(day < seeded.days_ago(130) for day in created_days)
        assert any(day > seeded.days_ago(110) for day in created_days)

    def test_e_exactly_three_open_report_tasks_all_in_client_work(self, seeded):
        reports = [task for task in seeded.tasks if "report" in task.title.lower()]

        assert sorted(task.title for task in reports) == REPORT_TITLES
        assert all(task.status is TaskStatus.OPEN for task in reports)
        assert {seeded.project_of(task) for task in reports} == {"Client work"}

    def test_e_the_word_report_appears_nowhere_else(self, seeded):
        for task in seeded.tasks:
            assert "report" not in (task.notes or "").lower(), task.title
        for project in seeded.workspace.projects:
            assert "report" not in f"{project.name} {project.description}".lower()

    def test_f_something_was_finished_on_every_second_day_of_the_last_two_weeks(self, seeded):
        completion_days = Counter(
            seeded.day(task.completed_at)
            for task in seeded.tasks
            if task.status is TaskStatus.DONE
        )

        for days in (1, 3, 5, 7, 9, 11, 13):
            assert completion_days[seeded.days_ago(days)] >= 1, f"nothing finished {days} days ago"

    def test_f_at_least_six_tasks_were_finished_in_the_previous_calendar_month(self, seeded):
        last_of_previous = seeded.anchor.replace(day=1) - timedelta(days=1)
        first_of_previous = last_of_previous.replace(day=1)

        finished = [
            task
            for task in seeded.tasks
            if task.status is TaskStatus.DONE
            and first_of_previous <= seeded.day(task.completed_at) <= last_of_previous
        ]

        assert len(finished) >= 6

    def test_g_book_dentist_appointment_is_open_unfiled_and_has_no_due_date(self, seeded):
        task = seeded.titled("Book dentist appointment")

        assert task.status is TaskStatus.OPEN
        assert task.project_id is None
        assert task.due_at is None
        assert task.priority is TaskPriority.NORMAL

    def test_g_order_kitchen_tiles_is_open_in_home_renovation_and_due_in_the_future(self, seeded):
        task = seeded.titled("Order kitchen tiles")

        assert task.status is TaskStatus.OPEN
        assert seeded.project_of(task) == "Home renovation"
        assert seeded.day(task.due_at) > seeded.anchor
        assert task.priority is TaskPriority.NORMAL

    def test_g_renew_domain_name_is_open_in_client_work_and_due_in_the_future(self, seeded):
        task = seeded.titled("Renew domain name")

        assert task.status is TaskStatus.OPEN
        assert seeded.project_of(task) == "Client work"
        assert seeded.day(task.due_at) > seeded.anchor

    def test_h_no_title_appears_twice(self, seeded):
        titles = Counter(task.title.casefold() for task in seeded.tasks)

        assert [title for title, count in titles.items() if count > 1] == []

    def test_i_words_the_walkthroughs_rely_on_being_absent_or_unique(self, seeded):
        titles = [task.title.lower() for task in seeded.tasks]
        texts = titles + [(task.notes or "").lower() for task in seeded.tasks]

        assert not any("tax" in text for text in texts)
        assert sum(1 for title in titles if "dentist" in title) == 1
        assert sum(1 for title in titles if "kitchen tiles" in title) == 1
        assert not any("plumber" in title for title in titles)

    def test_every_row_is_a_valid_entity_filed_under_a_seeded_project(self, seeded):
        project_ids = {project.project_id for project in seeded.workspace.projects}

        for task in seeded.tasks:
            task.check_invariants()
            assert task.project_id is None or task.project_id in project_ids
            assert task.updated_at >= task.created_at
        assert len({task.task_id for task in seeded.tasks}) == len(seeded.tasks)


class TestDemoScenarios:
    def test_the_word_home_fits_two_projects_and_a_full_name_fits_one(self):
        projects = build_demo_workspace(OWNER, MONDAY).projects

        ambiguous = ProjectReferenceResolver.resolve(projects, "Home")
        exact = ProjectReferenceResolver.resolve(projects, "home office")

        assert isinstance(ambiguous, Ambiguous)
        assert sorted(project.name for project in ambiguous.candidates) == [
            "Home office", "Home renovation"
        ]
        assert isinstance(exact, Resolved) and exact.match.name == "Home office"

    def test_the_word_report_fits_three_tasks_and_a_full_title_fits_one(self):
        tasks = build_demo_workspace(OWNER, MONDAY).tasks

        ambiguous = TaskReferenceResolver.resolve(tasks, "report")
        exact = TaskReferenceResolver.resolve(tasks, "expense report")

        assert isinstance(ambiguous, Ambiguous)
        assert sorted(task.title for task in ambiguous.candidates) == REPORT_TITLES
        assert isinstance(exact, Resolved) and exact.match.title == "Expense report"


class TestDeterminism:
    def test_the_same_owner_and_anchor_give_an_identical_dataset(self):
        first = build_demo_workspace(OWNER, MONDAY)
        second = build_demo_workspace(OWNER, MONDAY)

        assert first == second
        assert first.anchor_date == MONDAY
        assert build_demo_workspace(OWNER, MONDAY, DEMO_SEED, timezone=UTC) == first

    def test_the_dataset_does_not_depend_on_the_interpreters_hash_seed(self):
        script = (
            "import hashlib\n"
            "from datetime import date\n"
            "from src.shared.domain.value_objects.user_id import UserId\n"
            "from src.tasks.application.seed_demo_workspace import build_demo_workspace\n"
            f"owner = UserId.from_string('{OWNER}')\n"
            f"workspace = build_demo_workspace(owner, date.fromisoformat('{MONDAY.isoformat()}'))\n"
            "print(hashlib.sha256(repr(workspace).encode()).hexdigest())\n"
        )

        def digest_with(hash_seed: str) -> str:
            environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
            environment["PYTHONPATH"] = str(PROJECT_ROOT)
            completed = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                cwd=PROJECT_ROOT,
                env=environment,
            )
            return completed.stdout.strip()

        in_process = hashlib.sha256(repr(build_demo_workspace(OWNER, MONDAY)).encode()).hexdigest()
        assert digest_with("1") == digest_with("2") == in_process

    def test_different_owners_get_the_same_content_under_disjoint_ids(self):
        mine = build_demo_workspace(OWNER, MONDAY)
        theirs = build_demo_workspace(OTHER_OWNER, MONDAY)

        my_ids = {p.project_id for p in mine.projects} | {t.task_id for t in mine.tasks}
        their_ids = {p.project_id for p in theirs.projects} | {t.task_id for t in theirs.tasks}
        assert my_ids.isdisjoint(their_ids)
        assert [task.title for task in mine.tasks] == [task.title for task in theirs.tasks]
        assert [task.due_at for task in mine.tasks] == [task.due_at for task in theirs.tasks]
        assert all(task.user_id == OTHER_OWNER for task in theirs.tasks)
        assert all(project.user_id == OTHER_OWNER for project in theirs.projects)

    def test_another_anchor_moves_the_dates_and_keeps_the_ids(self):
        monday = build_demo_workspace(OWNER, MONDAY)
        a_week_later = build_demo_workspace(OWNER, MONDAY + timedelta(days=7))

        assert [task.task_id for task in monday.tasks] == [
            task.task_id for task in a_week_later.tasks
        ]
        assert [project.project_id for project in monday.projects] == [
            project.project_id for project in a_week_later.projects
        ]
        tiles_then = next(task for task in monday.tasks if task.title == "Order kitchen tiles")
        tiles_later = next(
            task for task in a_week_later.tasks if task.title == "Order kitchen tiles"
        )
        assert tiles_later.due_at - tiles_then.due_at == timedelta(days=7)
        assert tiles_later.created_at - tiles_then.created_at == timedelta(days=7)

    @pytest.mark.parametrize("seed", [1, 2, 3, 77, 2027])
    def test_the_drawn_rows_of_any_seed_stay_clear_of_what_the_pinned_rows_guarantee(self, seed):
        calendar = LocalCalendar(UTC)
        workspace = build_demo_workspace(OWNER, MONDAY, seed)
        first_quiet_day, last_quiet_day = (
            MONDAY - timedelta(days=130),
            MONDAY - timedelta(days=110),
        )

        assert workspace != build_demo_workspace(OWNER, MONDAY), "the seed drives the drawn rows"
        assert len({task.title for task in workspace.tasks}) == len(workspace.tasks)
        for task in workspace.tasks:
            for instant in (task.created_at, task.due_at, task.completed_at):
                if instant is not None:
                    assert not first_quiet_day <= calendar.day_of(instant) <= last_quiet_day
            assert calendar.day_of(task.created_at) >= MONDAY - timedelta(days=270)
            if task.completed_at is not None:
                assert task.created_at <= task.completed_at
                assert calendar.day_of(task.completed_at) <= MONDAY
        overdue = [
            task
            for task in workspace.tasks
            if task.status is TaskStatus.OPEN
            and task.due_at is not None
            and calendar.day_of(task.due_at) < MONDAY
        ]
        assert len(overdue) >= 5

    def test_an_anchor_too_close_to_the_edge_of_the_calendar_is_refused(self):
        with pytest.raises(ValidationError, match="too close to the edge"):
            build_demo_workspace(OWNER, LATEST_SUPPORTED_DAY)
        with pytest.raises(ValidationError, match="too close to the edge"):
            build_demo_workspace(OWNER, date(1971, 3, 1))


class TestSeedUseCase:
    async def test_writes_the_whole_workspace_in_one_transaction(self):
        repositories = InMemoryTasksRepositoryManager()

        summary = await SeedDemoWorkspaceUseCase(repositories).execute(OWNER, MONDAY)

        expected = build_demo_workspace(OWNER, MONDAY)
        assert repositories.transactions == 1
        assert await repositories.projects.list_by_user(OWNER) == sorted(
            expected.projects, key=lambda project: project.name
        )
        stored_tasks = await repositories.tasks.list_by_user(OWNER)
        assert sorted(stored_tasks, key=lambda task: task.title) == sorted(
            expected.tasks, key=lambda task: task.title
        )
        assert [kind for kind, _name in repositories.write_log] == (
            ["save_project"] * 4 + ["save_task"] * len(expected.tasks)
        ), "projects are written before the tasks filed under them"
        assert summary.tasks_count == len(expected.tasks)

    async def test_the_summary_describes_what_was_written(self):
        repositories = InMemoryTasksRepositoryManager()

        summary = await SeedDemoWorkspaceUseCase(repositories).execute(OWNER, MONDAY)

        tasks = await repositories.tasks.list_by_user(OWNER)
        count = Counter(task.status for task in tasks)
        assert isinstance(summary, SeedSummary)
        assert summary == SeedSummary(
            anchor_date=MONDAY,
            projects_count=4,
            tasks_count=len(tasks),
            open_count=count[TaskStatus.OPEN],
            done_count=count[TaskStatus.DONE],
            cancelled_count=count[TaskStatus.CANCELLED],
            overdue_count=sum(
                1
                for task in tasks
                if task.status is TaskStatus.OPEN
                and task.due_at is not None
                and task.due_at < LocalCalendar(UTC).start_of(MONDAY)
            ),
        )
        assert summary.open_count + summary.done_count + summary.cancelled_count == (
            summary.tasks_count
        )
        assert summary.overdue_count >= 5

    async def test_seeding_again_is_idempotent(self):
        repositories = InMemoryTasksRepositoryManager()
        use_case = SeedDemoWorkspaceUseCase(repositories)

        first_summary = await use_case.execute(OWNER, MONDAY)
        after_first = repositories.snapshot()
        second_summary = await use_case.execute(OWNER, MONDAY)

        assert repositories.snapshot() == after_first
        assert second_summary == first_summary
        assert repositories.transactions == 2

    async def test_seeding_on_a_later_day_refreshes_the_same_rows(self):
        repositories = InMemoryTasksRepositoryManager()
        use_case = SeedDemoWorkspaceUseCase(repositories)

        await use_case.execute(OWNER, MONDAY)
        _projects, tasks_then = repositories.snapshot()
        await use_case.execute(OWNER, MONDAY + timedelta(days=30))
        _projects, tasks_later = repositories.snapshot()

        assert set(tasks_later) == set(tasks_then), "no row is duplicated"
        assert tasks_later != tasks_then, "the dates follow the new anchor"

    async def test_seeding_touches_nothing_else_the_owner_or_anyone_has(self):
        repositories = InMemoryTasksRepositoryManager()
        my_project = a_project(OWNER, "Book club")
        my_task = a_task(OWNER, "Pick the next novel", project=my_project)
        their_task = a_task(OTHER_OWNER, "Quarterly report")
        await repositories.given(my_project, my_task, their_task)

        await SeedDemoWorkspaceUseCase(repositories).execute(OWNER, MONDAY)

        assert await repositories.projects.find_by_id(OWNER, my_project.project_id) == my_project
        assert await repositories.tasks.find_by_id(OWNER, my_task.task_id) == my_task
        assert await repositories.tasks.list_by_user(OTHER_OWNER) == [their_task]
        assert await repositories.projects.list_by_user(OTHER_OWNER) == []

    async def test_two_owners_can_be_seeded_side_by_side(self):
        repositories = InMemoryTasksRepositoryManager()
        use_case = SeedDemoWorkspaceUseCase(repositories)

        mine = await use_case.execute(OWNER, MONDAY)
        theirs = await use_case.execute(OTHER_OWNER, MONDAY)

        assert mine == theirs
        assert len(await repositories.tasks.list_by_user(OWNER)) == mine.tasks_count
        assert len(await repositories.tasks.list_by_user(OTHER_OWNER)) == theirs.tasks_count

    async def test_days_are_local_days_of_the_injected_zone(self):
        repositories = InMemoryTasksRepositoryManager()
        berlin = ZoneInfo("Europe/Berlin")

        summary = await SeedDemoWorkspaceUseCase(repositories, berlin).execute(OWNER, MONDAY)

        tasks = {task.title: task for task in await repositories.tasks.list_by_user(OWNER)}
        tiles = tasks["Order kitchen tiles"]
        assert tiles.due_at == datetime(2026, 9, 29, 22, 0, tzinfo=UTC), "midnight in Berlin"
        assert LocalCalendar(berlin).day_of(tiles.due_at) == date(2026, 9, 30)
        assert summary == await SeedDemoWorkspaceUseCase(
            InMemoryTasksRepositoryManager(), berlin
        ).execute(OWNER, MONDAY)

    async def test_a_refused_anchor_writes_nothing(self):
        repositories = InMemoryTasksRepositoryManager()

        with pytest.raises(ValidationError):
            await SeedDemoWorkspaceUseCase(repositories).execute(OWNER, LATEST_SUPPORTED_DAY)

        assert repositories.write_log == [] and repositories.transactions == 0
