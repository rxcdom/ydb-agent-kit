"""``ListProjectsUseCase``: the inventory by reference for the agent and by id for the API."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.list_projects import (
    ActivityTotals,
    ListProjectsUseCase,
    ProjectOverview,
)
from src.tasks.domain.value_objects.task_status import TaskStatus
from src.tasks.ports.task_repository import ProjectActivity
from tests.unit.tasks.builders import a_project, a_task, at
from tests.unit.tasks.envelopes import data_of
from tests.unit.tasks.in_memory import InMemoryTasksRepositoryManager

UTC = timezone.utc
AUCKLAND = ZoneInfo("Pacific/Auckland")

OWNER = UserId.generate()
STRANGER = UserId.generate()

NO_ACTIVITY = {"open_count": 0, "done_count": 0, "first_activity": None, "last_activity": None}


@pytest.fixture
async def repositories() -> InMemoryTasksRepositoryManager:
    manager = InMemoryTasksRepositoryManager()
    renovation = a_project(OWNER, "Home renovation", "Kitchen and bathroom remodel")
    garden = a_project(OWNER, "Garden")
    done, cancelled = TaskStatus.DONE, TaskStatus.CANCELLED
    await manager.given(
        renovation,
        garden,
        a_project(OWNER, "Client work", "Deliverables and invoices"),
        a_task(OWNER, "Order kitchen tiles", project=renovation, created_at=at(2026, 8, 3)),
        a_task(OWNER, "Fix leaking tap", project=renovation, created_at=at(2026, 8, 20)),
        a_task(
            OWNER, "Paint hallway", project=renovation, status=done,
            created_at=at(2026, 7, 14), completed_at=at(2026, 9, 5),
        ),
        a_task(
            OWNER, "Hire a skip", project=renovation, status=cancelled,
            created_at=at(2026, 6, 30),
        ),
        a_task(
            OWNER, "Plant herb seedlings", project=garden, status=done,
            created_at=at(2026, 5, 2), completed_at=at(2026, 5, 9),
        ),
        a_task(OWNER, "Book dentist appointment", created_at=at(2026, 9, 12)),
        a_task(
            OWNER, "Return library books", status=done,
            created_at=at(2026, 9, 1), completed_at=at(2026, 9, 14),
        ),
        a_project(STRANGER, "Garage"),
        a_task(STRANGER, "Sort the toolbox", created_at=at(2026, 1, 5)),
    )
    return manager


async def test_projects_come_by_name_with_counts_and_their_span_of_activity(repositories):
    outcome = await ListProjectsUseCase(repositories).execute(OWNER)

    assert data_of(outcome, "ok", ids=repositories.stored_ids()) == {
        "projects": [
            {"name": "Client work", "description": "Deliverables and invoices", **NO_ACTIVITY},
            {
                "name": "Garden",
                "description": None,
                "open_count": 0,
                "done_count": 1,
                "first_activity": "2026-05-02",
                "last_activity": "2026-05-09",
            },
            {
                "name": "Home renovation",
                "description": "Kitchen and bathroom remodel",
                "open_count": 2,
                "done_count": 1,
                "first_activity": "2026-06-30",
                "last_activity": "2026-09-05",
            },
        ],
        "unfiled": {
            "open_count": 1,
            "done_count": 1,
            "first_activity": "2026-09-01",
            "last_activity": "2026-09-14",
        },
    }


async def test_a_cancelled_task_is_activity_but_neither_open_nor_done(repositories):
    data = data_of(await ListProjectsUseCase(repositories).execute(OWNER), "ok")

    renovation = data["projects"][2]
    assert renovation["first_activity"] == "2026-06-30", "the cancelled task is the oldest one"
    assert renovation["open_count"] + renovation["done_count"] == 3, "four tasks, one cancelled"


async def test_an_owner_without_projects_and_tasks_has_no_data():
    repositories = InMemoryTasksRepositoryManager()
    await repositories.given(a_project(STRANGER, "Garage"), a_task(STRANGER, "Sort the toolbox"))

    outcome = await ListProjectsUseCase(repositories).execute(OWNER)

    assert data_of(outcome, "no_data", ids=repositories.stored_ids()) == {"projects_count": 0}


async def test_projects_without_any_task_are_still_an_inventory():
    repositories = InMemoryTasksRepositoryManager()
    await repositories.given(a_project(OWNER, "Garden"), a_project(OWNER, "Client work"))

    outcome = await ListProjectsUseCase(repositories).execute(OWNER)

    assert data_of(outcome, "ok") == {
        "projects": [
            {"name": "Client work", "description": None, **NO_ACTIVITY},
            {"name": "Garden", "description": None, **NO_ACTIVITY},
        ],
        "unfiled": NO_ACTIVITY,
    }


async def test_unfiled_tasks_alone_are_reported_under_the_unfiled_entry():
    repositories = InMemoryTasksRepositoryManager()
    await repositories.given(a_task(OWNER, "Renew passport", created_at=at(2026, 9, 3)))

    outcome = await ListProjectsUseCase(repositories).execute(OWNER)

    assert data_of(outcome, "ok") == {
        "projects": [],
        "unfiled": {
            "open_count": 1,
            "done_count": 0,
            "first_activity": "2026-09-03",
            "last_activity": "2026-09-03",
        },
    }


async def test_two_projects_sharing_a_name_are_listed_separately():
    repositories = InMemoryTasksRepositoryManager()
    first, second = a_project(OWNER, "Archive", "Paper"), a_project(OWNER, "Archive", "Photos")
    await repositories.given(first, second, a_task(OWNER, "Scan old letters", project=first))

    data = data_of(await ListProjectsUseCase(repositories).execute(OWNER), "ok")

    assert [project["name"] for project in data["projects"]] == ["Archive", "Archive"]
    assert sorted(project["open_count"] for project in data["projects"]) == [0, 1]


async def test_another_owners_projects_and_tasks_are_not_counted(repositories):
    mine = data_of(await ListProjectsUseCase(repositories).execute(OWNER), "ok")
    theirs = data_of(await ListProjectsUseCase(repositories).execute(STRANGER), "ok")

    assert "Garage" not in [project["name"] for project in mine["projects"]]
    assert theirs == {
        "projects": [{"name": "Garage", "description": None, **NO_ACTIVITY}],
        "unfiled": {
            "open_count": 1,
            "done_count": 0,
            "first_activity": "2026-01-05",
            "last_activity": "2026-01-05",
        },
    }


async def test_activity_days_are_local_days_of_the_injected_zone():
    repositories = InMemoryTasksRepositoryManager()
    garden = a_project(OWNER, "Garden")
    midday_utc = datetime(2026, 9, 17, 12, 30, tzinfo=UTC)
    await repositories.given(
        garden, a_task(OWNER, "Mow the lawn", project=garden, created_at=midday_utc)
    )

    in_auckland = await ListProjectsUseCase(repositories, AUCKLAND).execute(OWNER)
    in_utc = await ListProjectsUseCase(repositories).execute(OWNER)

    assert data_of(in_auckland, "ok")["projects"][0]["first_activity"] == "2026-09-18"
    assert data_of(in_utc, "ok")["projects"][0]["first_activity"] == "2026-09-17"


async def test_the_id_addressed_face_returns_entities_with_their_totals(repositories):
    overviews = await ListProjectsUseCase(repositories).list_overviews(OWNER)

    assert all(isinstance(overview, ProjectOverview) for overview in overviews)
    assert [overview.project.name for overview in overviews] == [
        "Client work", "Garden", "Home renovation"
    ]
    assert all(overview.project.user_id == OWNER for overview in overviews)
    renovation = overviews[2].totals
    assert (renovation.open_count, renovation.done_count) == (2, 1)
    assert renovation.first_activity == at(2026, 6, 30)
    assert renovation.last_activity == at(2026, 9, 5)
    assert overviews[0].totals == ActivityTotals()


async def test_the_id_addressed_face_of_an_owner_without_projects_is_an_empty_list():
    assert await ListProjectsUseCase(InMemoryTasksRepositoryManager()).list_overviews(OWNER) == []


def test_totals_fold_the_per_status_aggregates_of_one_project():
    totals = ActivityTotals.of(
        [
            ProjectActivity(None, TaskStatus.OPEN, 4, at(2026, 3, 1), at(2026, 3, 9)),
            ProjectActivity(None, TaskStatus.DONE, 2, at(2026, 2, 1), at(2026, 3, 20)),
            ProjectActivity(None, TaskStatus.CANCELLED, 7, at(2026, 1, 15), at(2026, 1, 16)),
        ]
    )

    assert totals == ActivityTotals(
        open_count=4, done_count=2, first_activity=at(2026, 1, 15), last_activity=at(2026, 3, 20)
    )
    assert ActivityTotals.of([]) == ActivityTotals()


async def test_listing_never_writes(repositories):
    before = repositories.snapshot()

    await ListProjectsUseCase(repositories).execute(OWNER)
    await ListProjectsUseCase(repositories).list_overviews(OWNER)

    assert repositories.write_log == []
    assert repositories.snapshot() == before
