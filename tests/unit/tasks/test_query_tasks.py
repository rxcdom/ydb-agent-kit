"""``QueryTasksUseCase``: one test per outcome of the vocabulary, plus the paged listing."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import pytest

from src.shared.domain.exceptions import ValidationError
from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.outcome import Outcome
from src.tasks.application.query_tasks import (
    DEFAULT_PAGE_SIZE,
    LIST_VIEW_CAP,
    MAX_PAGE_SIZE,
    QueryTasksUseCase,
    TaskPage,
)
from src.tasks.domain.entities.project import Project
from src.tasks.domain.exceptions import InvalidDateWindowError
from src.tasks.domain.value_objects.date_axis import DateAxis
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus
from src.tasks.ports.repository_manager import TasksRepositoryManager
from tests.unit.tasks.builders import a_project, a_task, at
from tests.unit.tasks.envelopes import data_of
from tests.unit.tasks.in_memory import InMemoryTasksRepositoryManager

UTC = timezone.utc
BERLIN = ZoneInfo("Europe/Berlin")

OWNER = UserId.generate()
STRANGER = UserId.generate()

CREATED_COVERAGE = {"from": "2026-07-01", "to": "2026-09-15"}
DUE_COVERAGE = {"from": "2026-09-04", "to": "2026-09-25"}
COMPLETED_COVERAGE = {"from": "2026-07-04", "to": "2026-09-12"}


@dataclass
class Workspace:
    repositories: InMemoryTasksRepositoryManager
    query: QueryTasksUseCase
    renovation: Project
    office: Project
    garden: Project

    def data(self, outcome: Outcome, status: str) -> Dict[str, Any]:
        return data_of(outcome, status, ids=self.repositories.stored_ids())


@pytest.fixture
async def workspace() -> Workspace:
    """Nine tasks of one owner over three projects, and a stranger whose rows must never show."""
    repositories = InMemoryTasksRepositoryManager()
    renovation = a_project(OWNER, "Home renovation", "Kitchen and bathroom remodel")
    office = a_project(OWNER, "Home office", "Desk setup and filing")
    garden = a_project(OWNER, "Garden")
    done = TaskStatus.DONE
    await repositories.given(
        renovation,
        office,
        garden,
        a_task(
            OWNER, "Order kitchen tiles", project=renovation,
            created_at=at(2026, 9, 1), due_at=at(2026, 9, 25),
        ),
        a_task(
            OWNER, "Fix leaking tap", project=renovation, priority=TaskPriority.HIGH,
            created_at=at(2026, 9, 3), due_at=at(2026, 9, 10),
        ),
        a_task(
            OWNER, "Paint hallway", project=renovation, status=done,
            created_at=at(2026, 8, 20), due_at=at(2026, 9, 4), completed_at=at(2026, 9, 5),
        ),
        a_task(
            OWNER, "Assemble desk", project=office, status=done, priority=TaskPriority.LOW,
            created_at=at(2026, 8, 25), completed_at=at(2026, 9, 12),
        ),
        a_task(
            OWNER, "Replace desk lamp", project=office, priority=TaskPriority.LOW,
            created_at=at(2026, 9, 8),
        ),
        a_task(
            OWNER, "Mow the lawn", project=garden, status=TaskStatus.CANCELLED,
            created_at=at(2026, 7, 10),
        ),
        a_task(
            OWNER, "Book dentist appointment", created_at=at(2026, 9, 15),
            notes="Ask about the night guard",
        ),
        a_task(
            OWNER, "Plant herb seedlings", project=garden, status=done,
            created_at=at(2026, 7, 1), completed_at=at(2026, 7, 4),
        ),
        a_task(OWNER, "Sand window frames", project=renovation, created_at=at(2026, 9, 10)),
        a_project(STRANGER, "Garage"),
        a_task(STRANGER, "Sort the toolbox", created_at=at(2026, 1, 5), due_at=at(2026, 12, 24)),
    )
    return Workspace(repositories, QueryTasksUseCase(repositories), renovation, office, garden)


def _group_counts(data: Dict[str, Any]) -> List[tuple]:
    return [(group["key"], group["count"]) for group in data["groups"]]


def _titles(data: Dict[str, Any]) -> List[str]:
    return [task["title"] for task in data["tasks"]]


class TestOk:
    async def test_a_summary_without_arguments_covers_everything_the_owner_has(self, workspace):
        outcome = await workspace.query.execute(OWNER)

        assert workspace.data(outcome, "ok") == {
            "window_used": CREATED_COVERAGE,
            "coverage": CREATED_COVERAGE,
            "date_field": "created",
            "total_count": 9,
            "excluded_without_date": 0,
            "status_counts": {"open": 5, "done": 3, "cancelled": 1},
        }

    async def test_window_used_is_the_named_window_and_coverage_stays_the_axis_span(
        self, workspace
    ):
        outcome = await workspace.query.execute(
            OWNER, date_from="2026-09-01", date_to="2026-09-30"
        )

        data = workspace.data(outcome, "ok")
        assert data["window_used"] == {"from": "2026-09-01", "to": "2026-09-30"}
        assert data["coverage"] == CREATED_COVERAGE
        assert data["total_count"] == 5
        assert data["status_counts"] == {"open": 5, "done": 0, "cancelled": 0}

    async def test_an_open_bound_is_reported_as_the_coverage_bound_it_defaulted_to(
        self, workspace
    ):
        outcome = await workspace.query.execute(OWNER, date_from="2026-09-01")

        assert workspace.data(outcome, "ok")["window_used"] == {
            "from": "2026-09-01", "to": "2026-09-15"
        }

    async def test_both_window_bounds_are_inclusive(self, workspace):
        outcome = await workspace.query.execute(
            OWNER, date_from="2026-09-03", date_to="2026-09-08", view="list"
        )

        assert _titles(workspace.data(outcome, "ok")) == ["Fix leaking tap", "Replace desk lamp"]

    async def test_without_group_by_a_summary_carries_no_groups(self, workspace):
        data = workspace.data(await workspace.query.execute(OWNER, group_by="none"), "ok")

        assert "groups" not in data and "tasks" not in data

    async def test_group_by_project_labels_groups_by_name_and_puts_unfiled_last(self, workspace):
        outcome = await workspace.query.execute(OWNER, group_by="project")

        assert workspace.data(outcome, "ok")["groups"] == [
            {"key": "Garden", "count": 2, "status_counts": {"open": 0, "done": 1, "cancelled": 1}},
            {
                "key": "Home office",
                "count": 2,
                "status_counts": {"open": 1, "done": 1, "cancelled": 0},
            },
            {
                "key": "Home renovation",
                "count": 4,
                "status_counts": {"open": 3, "done": 1, "cancelled": 0},
            },
            {"key": None, "count": 1, "status_counts": {"open": 1, "done": 0, "cancelled": 0}},
        ]

    async def test_group_by_status_follows_the_lifecycle_order(self, workspace):
        data = workspace.data(await workspace.query.execute(OWNER, group_by="status"), "ok")

        assert _group_counts(data) == [("open", 5), ("done", 3), ("cancelled", 1)]
        assert data["groups"][1]["status_counts"] == {"open": 0, "done": 3, "cancelled": 0}

    async def test_group_by_priority_runs_from_low_to_high(self, workspace):
        data = workspace.data(await workspace.query.execute(OWNER, group_by="priority"), "ok")

        assert _group_counts(data) == [("low", 2), ("normal", 6), ("high", 1)]
        assert data["groups"][1]["status_counts"] == {"open": 3, "done": 2, "cancelled": 1}

    async def test_group_by_week_keys_each_group_by_its_monday(self, workspace):
        data = workspace.data(await workspace.query.execute(OWNER, group_by="week"), "ok")

        assert _group_counts(data) == [
            ("2026-06-29", 1),
            ("2026-07-06", 1),
            ("2026-08-17", 1),
            ("2026-08-24", 1),
            ("2026-08-31", 2),
            ("2026-09-07", 2),
            ("2026-09-14", 1),
        ]

    async def test_group_by_month(self, workspace):
        data = workspace.data(await workspace.query.execute(OWNER, group_by="month"), "ok")

        assert _group_counts(data) == [("2026-07", 2), ("2026-08", 2), ("2026-09", 5)]
        assert sum(count for _key, count in _group_counts(data)) == data["total_count"]

    async def test_time_groups_follow_the_chosen_axis(self, workspace):
        outcome = await workspace.query.execute(OWNER, date_field="completed", group_by="month")

        data = workspace.data(outcome, "ok")
        assert data["date_field"] == "completed"
        assert _group_counts(data) == [("2026-07", 1), ("2026-09", 2)]

    async def test_two_projects_sharing_a_name_stay_two_groups(self, workspace):
        for _copy in range(2):
            archive = a_project(OWNER, "Archive")
            letters = a_task(OWNER, "Scan old letters", project=archive, created_at=at(2026, 9, 2))
            await workspace.repositories.given(archive, letters)

        data = workspace.data(await workspace.query.execute(OWNER, group_by="project"), "ok")

        assert _group_counts(data)[:2] == [("Archive", 1), ("Archive", 1)]

    @pytest.mark.parametrize(
        "reference, name",
        [("Garden", "Garden"), ("gArDeN", "Garden"), ("renov", "Home renovation"),
         ("filing", "Home office")],
    )
    async def test_a_resolved_project_is_named_in_the_envelope(self, workspace, reference, name):
        outcome = await workspace.query.execute(OWNER, project=reference, view="list")

        data = workspace.data(outcome, "ok")
        assert data["project_resolved"] == {"name": name}
        assert {task["project"] for task in data["tasks"]} == {name}
        assert data["total_count"] == len(data["tasks"])

    async def test_without_a_project_argument_nothing_is_reported_as_resolved(self, workspace):
        assert "project_resolved" not in workspace.data(await workspace.query.execute(OWNER), "ok")

    async def test_a_list_view_renders_names_and_local_days_oldest_first(self, workspace):
        outcome = await workspace.query.execute(OWNER, project="Home renovation", view="list")

        data = workspace.data(outcome, "ok")
        assert data["tasks"][0] == {
            "title": "Paint hallway",
            "project": "Home renovation",
            "status": "done",
            "priority": "normal",
            "due_at": "2026-09-04",
            "completed_at": "2026-09-05",
            "created_at": "2026-08-20",
        }
        assert _titles(data) == [
            "Paint hallway", "Order kitchen tiles", "Fix leaking tap", "Sand window frames"
        ]
        assert data["total_count"] == 4
        assert data["truncated"] is False
        assert "groups" not in data and "status_counts" not in data

    async def test_an_unfiled_task_is_listed_with_no_project(self, workspace):
        outcome = await workspace.query.execute(OWNER, text="dentist", view="list")

        (task,) = workspace.data(outcome, "ok")["tasks"]
        assert task["project"] is None and task["due_at"] is None and task["completed_at"] is None

    async def test_a_list_is_ordered_by_the_chosen_axis(self, workspace):
        outcome = await workspace.query.execute(OWNER, date_field="due", view="list")

        assert _titles(workspace.data(outcome, "ok")) == [
            "Paint hallway", "Fix leaking tap", "Order kitchen tiles"
        ]

    async def test_group_by_has_no_effect_in_a_list_view(self, workspace):
        outcome = await workspace.query.execute(OWNER, view="list", group_by="project")

        data = workspace.data(outcome, "ok")
        assert "groups" not in data
        assert len(data["tasks"]) == 9

    async def test_a_long_list_is_capped_and_says_so(self, workspace):
        await workspace.repositories.given(
            *[
                a_task(OWNER, f"Label storage box {number}", created_at=at(2026, 8, 1, 8, number))
                for number in range(60)
            ]
        )

        listed = workspace.data(await workspace.query.execute(OWNER, view="list"), "ok")
        summarised = workspace.data(await workspace.query.execute(OWNER), "ok")

        assert listed["total_count"] == 69
        assert len(listed["tasks"]) == LIST_VIEW_CAP
        assert listed["truncated"] is True
        assert sum(summarised["status_counts"].values()) == 69, "a summary counts every row"

    async def test_text_matches_titles_and_notes_whatever_the_case(self, workspace):
        by_title = await workspace.query.execute(OWNER, text="DESK", view="list")
        by_notes = await workspace.query.execute(OWNER, text="Night Guard", view="list")

        assert _titles(workspace.data(by_title, "ok")) == ["Assemble desk", "Replace desk lamp"]
        assert _titles(workspace.data(by_notes, "ok")) == ["Book dentist appointment"]

    async def test_filters_combine(self, workspace):
        outcome = await workspace.query.execute(
            OWNER,
            project="Home renovation",
            statuses=["open"],
            priority=["normal", "high"],
            date_from="2026-09-02",
            view="list",
        )

        assert _titles(workspace.data(outcome, "ok")) == ["Fix leaking tap", "Sand window frames"]

    async def test_closed_vocabularies_are_read_case_insensitively(self, workspace):
        outcome = await workspace.query.execute(
            OWNER, date_field=" DUE ", statuses=["Open"], group_by="Status", view="SUMMARY"
        )

        data = workspace.data(outcome, "ok")
        assert data["date_field"] == "due"
        assert _group_counts(data) == [("open", 2)]

    async def test_another_owners_rows_never_show(self, workspace):
        everything = workspace.data(await workspace.query.execute(OWNER, view="list"), "ok")
        theirs = workspace.data(await workspace.query.execute(STRANGER, view="list"), "ok")

        assert "Sort the toolbox" not in _titles(everything)
        assert _titles(theirs) == ["Sort the toolbox"]
        assert theirs["coverage"] == {"from": "2026-01-05", "to": "2026-01-05"}


class TestLocalDays:
    async def test_days_are_local_days_of_the_injected_zone(self):
        repositories = InMemoryTasksRepositoryManager()
        late_evening_utc = datetime(2026, 9, 17, 22, 30, tzinfo=UTC)
        await repositories.given(a_task(OWNER, "Call the roofer", created_at=late_evening_utc))
        window = {"date_from": "2026-09-18", "date_to": "2026-09-18", "view": "list"}

        in_berlin = await QueryTasksUseCase(repositories, BERLIN).execute(OWNER, **window)
        in_utc = await QueryTasksUseCase(repositories).execute(OWNER, **window)

        data = data_of(in_berlin, "ok")
        assert data["coverage"] == {"from": "2026-09-18", "to": "2026-09-18"}
        assert data["tasks"][0]["created_at"] == "2026-09-18"
        assert data_of(in_utc, "coverage_gap")["coverage"] == {
            "from": "2026-09-17", "to": "2026-09-17"
        }

    async def test_a_date_only_due_day_is_found_on_that_local_day(self):
        repositories = InMemoryTasksRepositoryManager()
        local_midnight = datetime(2026, 9, 24, 22, 0, tzinfo=UTC)
        await repositories.given(a_task(OWNER, "Renew domain name", due_at=local_midnight))

        outcome = await QueryTasksUseCase(repositories, BERLIN).execute(
            OWNER, date_field="due", date_from="2026-09-25", date_to="2026-09-25", view="list"
        )

        assert data_of(outcome, "ok")["tasks"][0]["due_at"] == "2026-09-25"

    async def test_weeks_are_cut_at_local_midnight(self):
        repositories = InMemoryTasksRepositoryManager()
        sunday_evening_utc = datetime(2026, 9, 20, 22, 30, tzinfo=UTC)
        await repositories.given(a_task(OWNER, "Plan the week", created_at=sunday_evening_utc))

        in_berlin = await QueryTasksUseCase(repositories, BERLIN).execute(OWNER, group_by="week")
        in_utc = await QueryTasksUseCase(repositories).execute(OWNER, group_by="week")

        assert _group_counts(data_of(in_berlin, "ok")) == [("2026-09-21", 1)]
        assert _group_counts(data_of(in_utc, "ok")) == [("2026-09-14", 1)]


class TestNoData:
    async def test_an_owner_without_tasks_gets_no_data_with_the_project_count(self):
        repositories = InMemoryTasksRepositoryManager()
        await repositories.given(
            a_project(OWNER, "Garden"),
            a_project(OWNER, "Home office"),
            a_project(STRANGER, "Garage"),
            a_task(STRANGER, "Sort the toolbox"),
        )

        outcome = await QueryTasksUseCase(repositories).execute(OWNER)

        assert data_of(outcome, "no_data", ids=repositories.stored_ids()) == {"projects_count": 2}

    async def test_no_data_wins_over_every_window_and_filter(self):
        outcome = await QueryTasksUseCase(InMemoryTasksRepositoryManager()).execute(
            OWNER, date_from="2026-09-01", project="Garden", statuses=["open"], text="tiles"
        )

        assert data_of(outcome, "no_data") == {"projects_count": 0}


class TestCoverageGap:
    async def test_a_window_entirely_before_the_coverage(self, workspace):
        outcome = await workspace.query.execute(OWNER, date_to="2026-06-30")

        data = workspace.data(outcome, "coverage_gap")
        assert data["coverage"] == CREATED_COVERAGE
        assert data["requested_window"] == {"from": None, "to": "2026-06-30"}
        assert data["date_field"] == "created"
        assert set(data) == {"coverage", "requested_window", "date_field", "note"}

    async def test_a_window_entirely_after_the_coverage(self, workspace):
        outcome = await workspace.query.execute(
            OWNER, date_from="2026-09-16", date_to="2026-09-30"
        )

        data = workspace.data(outcome, "coverage_gap")
        assert data["requested_window"] == {"from": "2026-09-16", "to": "2026-09-30"}

    async def test_coverage_is_computed_on_the_selected_axis(self, workspace):
        august = {"date_from": "2026-08-01", "date_to": "2026-08-31"}

        on_created = await workspace.query.execute(OWNER, **august)
        on_due = await workspace.query.execute(OWNER, date_field="due", **august)

        assert workspace.data(on_created, "ok")["total_count"] == 2
        data = workspace.data(on_due, "coverage_gap")
        assert data["coverage"] == DUE_COVERAGE
        assert data["date_field"] == "due"

    async def test_a_gap_is_reported_whatever_the_project_reference_says(self, workspace):
        outcome = await workspace.query.execute(OWNER, date_to="2026-06-30", project="Home")

        assert workspace.data(outcome, "coverage_gap")["coverage"] == CREATED_COVERAGE

    @pytest.mark.parametrize("axis", ["due", "completed"])
    async def test_an_axis_without_any_value_has_no_coverage(self, axis):
        repositories = InMemoryTasksRepositoryManager()
        await repositories.given(a_task(OWNER, "Mow the lawn"))

        outcome = await QueryTasksUseCase(repositories).execute(OWNER, date_field=axis)

        data = data_of(outcome, "coverage_gap")
        assert data["coverage"] is None
        assert data["requested_window"] == {"from": None, "to": None}
        assert axis in data["note"]


class TestEmptyFilter:
    async def test_an_unmatched_project_lists_the_projects_that_exist(self, workspace):
        outcome = await workspace.query.execute(OWNER, project="Garage")

        data = workspace.data(outcome, "empty_filter")
        assert data["available_projects"] == ["Garden", "Home office", "Home renovation"]
        assert data["count_without_filters"] == 9
        assert data["available_statuses"] == ["open", "done", "cancelled"]
        assert data["coverage"] == data["window_used"] == CREATED_COVERAGE
        assert "'Garage'" in data["note"]
        assert set(data) == {
            "coverage",
            "window_used",
            "count_without_filters",
            "available_projects",
            "available_statuses",
            "note",
        }

    async def test_a_status_that_matches_nothing_in_the_window(self, workspace):
        outcome = await workspace.query.execute(
            OWNER, statuses=["cancelled"], date_from="2026-09-01", date_to="2026-09-30"
        )

        data = workspace.data(outcome, "empty_filter")
        assert data["count_without_filters"] == 5
        assert data["available_statuses"] == ["open"]
        assert data["window_used"] == {"from": "2026-09-01", "to": "2026-09-30"}

    async def test_a_priority_that_matches_nothing_in_the_window(self, workspace):
        outcome = await workspace.query.execute(
            OWNER, priority=["high"], date_from="2026-08-01", date_to="2026-08-31"
        )

        data = workspace.data(outcome, "empty_filter")
        assert data["count_without_filters"] == 2
        assert data["available_statuses"] == ["done"]

    async def test_a_text_that_matches_nothing(self, workspace):
        outcome = await workspace.query.execute(OWNER, text="plumber")

        data = workspace.data(outcome, "empty_filter")
        assert data["count_without_filters"] == 9
        assert data["available_projects"] == ["Garden", "Home office", "Home renovation"]

    async def test_a_resolved_project_with_nothing_in_the_window(self, workspace):
        outcome = await workspace.query.execute(OWNER, project="Garden", date_from="2026-09-01")

        assert workspace.data(outcome, "empty_filter")["count_without_filters"] == 5

    async def test_count_without_filters_drops_every_filter_but_keeps_the_window(self, workspace):
        outcome = await workspace.query.execute(
            OWNER,
            project="Garden",
            statuses=["open"],
            priority=["high"],
            text="lawn",
            date_from="2026-07-01",
            date_to="2026-07-31",
        )

        data = workspace.data(outcome, "empty_filter")
        assert data["count_without_filters"] == 2
        assert data["available_statuses"] == ["done", "cancelled"]


class TestNoRecords:
    async def test_a_quiet_window_inside_the_coverage_is_an_honest_gap(self, workspace):
        outcome = await workspace.query.execute(
            OWNER, date_from="2026-07-15", date_to="2026-08-10"
        )

        assert workspace.data(outcome, "no_records") == {
            "coverage": CREATED_COVERAGE,
            "window_used": {"from": "2026-07-15", "to": "2026-08-10"},
            "count_without_window": 9,
        }

    async def test_filters_that_narrow_nothing_are_not_blamed(self, workspace):
        outcome = await workspace.query.execute(
            OWNER,
            date_from="2026-07-15",
            date_to="2026-08-10",
            project="  ",
            statuses=["open", "done", "cancelled"],
            priority=["low", "normal", "high"],
            text="",
        )

        assert workspace.data(outcome, "no_records")["count_without_window"] == 9

    async def test_count_without_window_is_counted_on_the_same_axis(self, workspace):
        outcome = await workspace.query.execute(
            OWNER, date_field="completed", date_from="2026-08-01", date_to="2026-08-31"
        )

        data = workspace.data(outcome, "no_records")
        assert data["coverage"] == COMPLETED_COVERAGE
        assert data["count_without_window"] == 3


class TestAmbiguousSource:
    async def test_a_reference_that_fits_two_projects_is_never_resolved_silently(self, workspace):
        outcome = await workspace.query.execute(OWNER, project="Home")

        data = workspace.data(outcome, "ambiguous_source")
        assert data["reference"] == "Home"
        assert data["reference_kind"] == "project"
        assert data["matched"] == [
            {"title_or_name": "Home office", "description": "Desk setup and filing"},
            {"title_or_name": "Home renovation", "description": "Kitchen and bathroom remodel"},
        ]
        assert "2 projects" in data["note"]
        assert set(data) == {"reference", "reference_kind", "matched", "note"}

    async def test_an_exact_project_name_ends_the_ambiguity(self, workspace):
        home = a_project(OWNER, "Home")
        await workspace.repositories.given(
            home, a_task(OWNER, "Clean the windows", project=home, created_at=at(2026, 9, 2))
        )

        outcome = await workspace.query.execute(OWNER, project="home")

        data = workspace.data(outcome, "ok")
        assert data["project_resolved"] == {"name": "Home"}
        assert data["total_count"] == 1


class _UnreadableRepositories(TasksRepositoryManager):
    """Fails the test the moment a use case reaches for data."""

    @property
    def projects(self):
        raise AssertionError("projects were read for a call that is invalid on its face")

    @property
    def tasks(self):
        raise AssertionError("tasks were read for a call that is invalid on its face")

    async def execute_in_transaction(self, operations):
        raise AssertionError("a transaction was opened for a read")


class TestFilterError:
    @pytest.mark.parametrize("statuses", [["open"], ["cancelled"], ["open", "cancelled"]])
    async def test_the_completed_axis_contradicts_statuses_that_exclude_done(
        self, workspace, statuses
    ):
        outcome = await workspace.query.execute(OWNER, date_field="completed", statuses=statuses)

        data = workspace.data(outcome, "filter_error")
        assert data["error_code"] == "completed_axis_excludes_statuses"
        assert "done" in data["message"] and "date_field" in data["message"]
        assert set(data) == {"error_code", "message"}

    @pytest.mark.parametrize("statuses", [None, [], ["done"], ["open", "done"]])
    async def test_the_completed_axis_accepts_statuses_that_include_done(
        self, workspace, statuses
    ):
        outcome = await workspace.query.execute(OWNER, date_field="completed", statuses=statuses)

        data = workspace.data(outcome, "ok")
        assert data["total_count"] == 3
        assert data["status_counts"] == {"open": 0, "done": 3, "cancelled": 0}

    @pytest.mark.parametrize(
        "date_from, date_to, error_code",
        [
            ("18.09.2026", None, "malformed_date"),
            (None, "2026-02-30", "malformed_date"),
            ("last week", None, "malformed_date"),
            ("2026-09-30", "2026-09-01", "inverted_window"),
            ("1969-01-01", None, "date_out_of_range"),
        ],
    )
    async def test_malformed_and_inverted_dates(self, workspace, date_from, date_to, error_code):
        outcome = await workspace.query.execute(OWNER, date_from=date_from, date_to=date_to)

        data = workspace.data(outcome, "filter_error")
        assert data["error_code"] == error_code
        assert data["message"]

    @pytest.mark.parametrize(
        "arguments, argument_name, allowed",
        [
            ({"date_field": "updated"}, "date_field", "created, due, completed"),
            ({"group_by": "year"}, "group_by", "none, project, status, priority, week, month"),
            ({"view": "table"}, "view", "summary, list"),
            ({"statuses": ["open", "paused"]}, "statuses", "open, done, cancelled"),
            ({"priority": ["urgent"]}, "priority", "low, normal, high"),
        ],
    )
    async def test_values_outside_a_closed_vocabulary(
        self, workspace, arguments, argument_name, allowed
    ):
        outcome = await workspace.query.execute(OWNER, **arguments)

        data = workspace.data(outcome, "filter_error")
        assert data["error_code"] == "unknown_value"
        assert argument_name in data["message"] and allowed in data["message"]

    @pytest.mark.parametrize(
        "arguments",
        [
            {"date_from": "soon"},
            {"date_from": "2026-09-30", "date_to": "2026-09-01"},
            {"date_field": "completed", "statuses": ["open"]},
            {"view": "table"},
        ],
    )
    async def test_a_bad_call_is_answered_before_any_data_is_read(self, arguments):
        outcome = await QueryTasksUseCase(_UnreadableRepositories()).execute(OWNER, **arguments)

        assert data_of(outcome, "filter_error")["error_code"]


class TestExcludedWithoutDate:
    async def test_the_due_axis_reports_how_many_tasks_it_left_out(self, workspace):
        outcome = await workspace.query.execute(OWNER, date_field="due")

        data = workspace.data(outcome, "ok")
        assert data["coverage"] == data["window_used"] == DUE_COVERAGE
        assert data["total_count"] == 3
        assert data["excluded_without_date"] == 6
        assert data["total_count"] + data["excluded_without_date"] == 9

    async def test_the_left_out_count_respects_the_narrowing_filters(self, workspace):
        in_project = await workspace.query.execute(
            OWNER, date_field="due", project="Home renovation"
        )
        open_only = await workspace.query.execute(OWNER, date_field="due", statuses=["open"])

        assert workspace.data(in_project, "ok")["total_count"] == 3
        assert workspace.data(in_project, "ok")["excluded_without_date"] == 1
        assert workspace.data(open_only, "ok")["total_count"] == 2
        assert workspace.data(open_only, "ok")["excluded_without_date"] == 3

    async def test_the_window_does_not_apply_to_tasks_without_a_due_date(self, workspace):
        overdue = await workspace.query.execute(
            OWNER, date_field="due", date_to="2026-09-17", statuses=["open"], view="list"
        )

        data = workspace.data(overdue, "ok")
        assert _titles(data) == ["Fix leaking tap"]
        assert data["excluded_without_date"] == 3

    async def test_an_unfinished_task_is_not_missing_a_completion_date(self, workspace):
        outcome = await workspace.query.execute(OWNER, date_field="completed")

        data = workspace.data(outcome, "ok")
        assert data["coverage"] == COMPLETED_COVERAGE
        assert data["total_count"] == 3
        assert data["excluded_without_date"] == 0

    async def test_every_task_has_a_creation_date(self, workspace):
        outcome = await workspace.query.execute(OWNER, date_field="created")

        assert workspace.data(outcome, "ok")["excluded_without_date"] == 0


class TestListPage:
    async def test_pages_run_newest_first_and_say_whether_more_exist(self, workspace):
        first = await workspace.query.list_page(OWNER, limit=4)
        second = await workspace.query.list_page(OWNER, limit=4, offset=4)
        last = await workspace.query.list_page(OWNER, limit=4, offset=8)

        assert isinstance(first, TaskPage)
        assert [task.title for task in first.tasks] == [
            "Book dentist appointment", "Sand window frames", "Replace desk lamp", "Fix leaking tap"
        ]
        assert [task.title for task in second.tasks] == [
            "Order kitchen tiles", "Assemble desk", "Paint hallway", "Mow the lawn"
        ]
        assert [task.title for task in last.tasks] == ["Plant herb seedlings"]
        assert (first.total_count, second.total_count, last.total_count) == (9, 9, 9)
        assert (first.has_more, second.has_more, last.has_more) == (True, True, False)

    async def test_an_offset_past_the_end_is_an_empty_last_page(self, workspace):
        page = await workspace.query.list_page(OWNER, offset=9)

        assert page == TaskPage(tasks=[], total_count=9, has_more=False)

    async def test_the_default_page_holds_everything_here(self, workspace):
        page = await workspace.query.list_page(OWNER)

        assert DEFAULT_PAGE_SIZE == 50
        assert len(page.tasks) == 9 and page.has_more is False
        assert all(task.user_id == OWNER for task in page.tasks), "entities, addressed by id"

    async def test_status_and_project_filters(self, workspace):
        done = await workspace.query.list_page(OWNER, status=TaskStatus.DONE)
        in_garden = await workspace.query.list_page(OWNER, project_id=workspace.garden.project_id)
        both = await workspace.query.list_page(
            OWNER, status=TaskStatus.DONE, project_id=workspace.garden.project_id
        )

        assert {task.title for task in done.tasks} == {
            "Paint hallway", "Assemble desk", "Plant herb seedlings"
        }
        assert {task.title for task in in_garden.tasks} == {"Mow the lawn", "Plant herb seedlings"}
        assert [task.title for task in both.tasks] == ["Plant herb seedlings"]
        assert (done.total_count, in_garden.total_count, both.total_count) == (3, 2, 1)

    async def test_the_window_is_inclusive_on_the_chosen_axis(self, workspace):
        created = await workspace.query.list_page(
            OWNER, date_from=date(2026, 9, 3), date_to=date(2026, 9, 8)
        )
        due = await workspace.query.list_page(OWNER, date_field=DateAxis.DUE)
        due_by_the_tenth = await workspace.query.list_page(
            OWNER, date_field=DateAxis.DUE, date_to=date(2026, 9, 10)
        )

        assert [task.title for task in created.tasks] == ["Replace desk lamp", "Fix leaking tap"]
        assert [task.title for task in due.tasks] == [
            "Order kitchen tiles", "Fix leaking tap", "Paint hallway"
        ], "tasks without a due date are not listed on the due axis"
        assert due.total_count == 3
        assert [task.title for task in due_by_the_tenth.tasks] == [
            "Fix leaking tap", "Paint hallway"
        ]

    async def test_the_window_uses_local_days_of_the_injected_zone(self):
        repositories = InMemoryTasksRepositoryManager()
        await repositories.given(
            a_task(OWNER, "Call the roofer", created_at=datetime(2026, 9, 17, 22, 30, tzinfo=UTC))
        )
        the_eighteenth = {"date_from": date(2026, 9, 18), "date_to": date(2026, 9, 18)}

        in_berlin = await QueryTasksUseCase(repositories, BERLIN).list_page(OWNER, **the_eighteenth)
        in_utc = await QueryTasksUseCase(repositories).list_page(OWNER, **the_eighteenth)

        assert (in_berlin.total_count, in_utc.total_count) == (1, 0)

    async def test_paging_arguments_are_validated(self, workspace):
        for bad_limit in (0, -1, MAX_PAGE_SIZE + 1):
            with pytest.raises(ValidationError, match="limit"):
                await workspace.query.list_page(OWNER, limit=bad_limit)
        with pytest.raises(ValidationError, match="offset"):
            await workspace.query.list_page(OWNER, offset=-1)

        assert len((await workspace.query.list_page(OWNER, limit=MAX_PAGE_SIZE)).tasks) == 9

    async def test_an_inverted_window_is_a_domain_error(self, workspace):
        with pytest.raises(InvalidDateWindowError):
            await workspace.query.list_page(
                OWNER, date_from=date(2026, 9, 30), date_to=date(2026, 9, 1)
            )

    async def test_another_owners_tasks_are_not_listed(self, workspace):
        mine = await workspace.query.list_page(OWNER)
        theirs = await workspace.query.list_page(STRANGER)
        their_view_of_my_project = await workspace.query.list_page(
            STRANGER, project_id=workspace.garden.project_id
        )

        assert "Sort the toolbox" not in {task.title for task in mine.tasks}
        assert [task.title for task in theirs.tasks] == ["Sort the toolbox"]
        assert their_view_of_my_project == TaskPage(tasks=[], total_count=0, has_more=False)


async def test_reads_never_write(workspace):
    before = workspace.repositories.snapshot()

    await workspace.query.execute(OWNER, group_by="week")
    await workspace.query.execute(OWNER, project="Home")
    await workspace.query.list_page(OWNER)

    assert workspace.repositories.write_log == []
    assert workspace.repositories.snapshot() == before
