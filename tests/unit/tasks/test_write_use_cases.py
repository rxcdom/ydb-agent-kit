"""Create, update and delete through both faces: by id for the API, by reference for the agent.

The guards that matter most are here: a write never acts on an ambiguous
reference, reports exactly what it changed, and never reaches another owner's row.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import pytest

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.create_task import CreateTaskUseCase
from src.tasks.application.delete_task import DeleteTaskUseCase
from src.tasks.application.outcome import Outcome
from src.tasks.application.update_task import TaskChangeRequest, TaskChanges, UpdateTaskUseCase
from src.tasks.domain.entities.project import Project
from src.tasks.domain.entities.task import MAX_TASK_NOTES_LENGTH, MAX_TASK_TITLE_LENGTH, Task
from src.tasks.domain.exceptions import (
    InvalidTaskError,
    ProjectNotFoundError,
    TaskAccessDeniedError,
    TaskNotFoundError,
)
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus
from tests.unit.tasks.builders import a_project, a_task, at
from tests.unit.tasks.envelopes import data_of
from tests.unit.tasks.in_memory import InMemoryTaskRepository, InMemoryTasksRepositoryManager

UTC = timezone.utc
BERLIN = ZoneInfo("Europe/Berlin")

OWNER = UserId.generate()
STRANGER = UserId.generate()

CLOCK_NOW = at(2026, 9, 18, 15, 45)
OWNER_PROJECT_NAMES = ["Client work", "Home office", "Home renovation"]


def _clock() -> datetime:
    return CLOCK_NOW


@dataclass
class Workspace:
    repositories: InMemoryTasksRepositoryManager
    create: CreateTaskUseCase
    update: UpdateTaskUseCase
    delete: DeleteTaskUseCase
    renovation: Project
    office: Project
    clients: Project
    quarterly: Task
    expense: Task
    weekly: Task
    paint_in_renovation: Task
    paint_in_office: Task
    dentist: Task
    hallway: Task
    tiles: Task
    their_clients: Project
    their_dentist: Task
    their_quarterly: Task
    their_laser: Task

    def data(self, outcome: Outcome, status: str) -> Dict[str, Any]:
        return data_of(outcome, status, ids=self.repositories.stored_ids())

    async def stored(self, task: Task, owner: UserId = OWNER) -> Optional[Task]:
        return await self.repositories.tasks.find_by_id(owner, task.task_id)

    def assert_untouched(self, before) -> None:
        assert self.repositories.write_log == [], "something was written"
        assert self.repositories.snapshot() == before, "a stored row differs"


async def _workspace(repositories: InMemoryTasksRepositoryManager, zone=UTC) -> Workspace:
    renovation = a_project(OWNER, "Home renovation", "Kitchen and bathroom remodel")
    office = a_project(OWNER, "Home office", "Desk setup and filing")
    clients = a_project(OWNER, "Client work", "Deliverables and invoices")
    their_clients = a_project(STRANGER, "Client work")
    rows = dict(
        quarterly=a_task(
            OWNER, "Quarterly report", project=clients, priority=TaskPriority.HIGH,
            created_at=at(2026, 9, 2), due_at=at(2026, 9, 30, 0),
        ),
        expense=a_task(
            OWNER, "Expense report", project=clients,
            created_at=at(2026, 9, 3), due_at=at(2026, 9, 14, 0),
        ),
        weekly=a_task(OWNER, "Weekly status report", project=clients, created_at=at(2026, 9, 4)),
        paint_in_renovation=a_task(
            OWNER, "Buy paint", project=renovation, created_at=at(2026, 9, 5)
        ),
        paint_in_office=a_task(OWNER, "Buy paint", project=office, created_at=at(2026, 9, 6)),
        dentist=a_task(
            OWNER, "Book dentist appointment", created_at=at(2026, 9, 7),
            notes="Ask about the night guard",
        ),
        hallway=a_task(
            OWNER, "Paint hallway ceiling", project=renovation, status=TaskStatus.DONE,
            created_at=at(2026, 9, 1), completed_at=at(2026, 9, 10),
        ),
        tiles=a_task(
            OWNER, "Order kitchen tiles", project=renovation,
            created_at=at(2026, 9, 8), due_at=at(2026, 9, 25, 9, 30), notes="Matte white",
        ),
        their_dentist=a_task(STRANGER, "Book dentist appointment", created_at=at(2026, 9, 7)),
        their_quarterly=a_task(
            STRANGER, "Quarterly report", project=their_clients, created_at=at(2026, 9, 2)
        ),
        their_laser=a_task(STRANGER, "Calibrate the laser", created_at=at(2026, 9, 9)),
    )
    await repositories.given(
        renovation, office, clients, their_clients, a_project(STRANGER, "Secret lab"),
        *rows.values(),
    )
    return Workspace(
        repositories=repositories,
        create=CreateTaskUseCase(repositories, zone, _clock),
        update=UpdateTaskUseCase(repositories, zone, _clock),
        delete=DeleteTaskUseCase(repositories, zone),
        renovation=renovation,
        office=office,
        clients=clients,
        their_clients=their_clients,
        **rows,
    )


@pytest.fixture
async def workspace() -> Workspace:
    return await _workspace(InMemoryTasksRepositoryManager())


@pytest.fixture
async def berlin_workspace() -> Workspace:
    return await _workspace(InMemoryTasksRepositoryManager(), BERLIN)


class _LeakyTaskRepository(InMemoryTaskRepository):
    """A broken repository that forgets the owner, to prove the use cases do not rely on it."""

    async def find_by_id(self, user_id: UserId, task_id: UUID) -> Optional[Task]:
        return self.snapshot().get(task_id)

    async def list_by_user(self, user_id: UserId):
        return list(self.snapshot().values())


class _LeakyRepositories(InMemoryTasksRepositoryManager):
    def __init__(self) -> None:
        super().__init__()
        self._tasks = _LeakyTaskRepository(self.write_log)


class TestCreateById:
    async def test_creates_an_open_task_with_the_defaults(self, workspace):
        task = await workspace.create.execute_by_id(OWNER, "  Descale the kettle  ")

        assert task.title == "Descale the kettle"
        assert (task.status, task.priority) == (TaskStatus.OPEN, TaskPriority.NORMAL)
        assert (task.project_id, task.due_at, task.completed_at, task.notes) == (None,) * 4
        assert task.created_at == task.updated_at == CLOCK_NOW
        assert task.user_id == OWNER
        assert await workspace.stored(task) == task
        assert workspace.repositories.write_log == [("save_task", "Descale the kettle")]

    async def test_every_field_can_be_given(self, workspace):
        task = await workspace.create.execute_by_id(
            OWNER,
            "Pick worktop material",
            project_id=workspace.renovation.project_id,
            due_at=date(2026, 10, 5),
            priority=TaskPriority.HIGH,
            notes="Oak or laminate",
        )

        stored = await workspace.stored(task)
        assert stored.project_id == workspace.renovation.project_id
        assert stored.priority is TaskPriority.HIGH
        assert stored.notes == "Oak or laminate"
        assert stored.due_at == at(2026, 10, 5, 0)

    async def test_a_date_only_due_day_is_the_local_midnight_that_starts_it(
        self, berlin_workspace
    ):
        task = await berlin_workspace.create.execute_by_id(
            OWNER, "Renew domain name", due_at=date(2026, 9, 25)
        )

        assert task.due_at == datetime(2026, 9, 24, 22, 0, tzinfo=UTC)

    async def test_a_moment_is_kept_and_a_naive_one_is_local_wall_clock_time(
        self, berlin_workspace
    ):
        plus_five = timezone(timedelta(hours=5))
        aware = await berlin_workspace.create.execute_by_id(
            OWNER, "Join kickoff call", due_at=datetime(2026, 9, 25, 14, 0, tzinfo=plus_five)
        )
        naive = await berlin_workspace.create.execute_by_id(
            OWNER, "Send meeting notes", due_at=datetime(2026, 9, 25, 14, 0)
        )

        assert aware.due_at == datetime(2026, 9, 25, 9, 0, tzinfo=UTC)
        assert naive.due_at == datetime(2026, 9, 25, 12, 0, tzinfo=UTC)

    async def test_an_unknown_project_id_is_not_found_and_nothing_is_written(self, workspace):
        before = workspace.repositories.snapshot()

        with pytest.raises(ProjectNotFoundError):
            await workspace.create.execute_by_id(OWNER, "Buy screws", project_id=uuid4())

        workspace.assert_untouched(before)

    async def test_another_owners_project_is_not_found_either(self, workspace):
        before = workspace.repositories.snapshot()

        with pytest.raises(ProjectNotFoundError):
            await workspace.create.execute_by_id(
                OWNER, "Buy screws", project_id=workspace.their_clients.project_id
            )

        workspace.assert_untouched(before)

    @pytest.mark.parametrize(
        "arguments",
        [
            {"title": "   "},
            {"title": "x" * (MAX_TASK_TITLE_LENGTH + 1)},
            {"title": "Buy screws", "notes": "x" * (MAX_TASK_NOTES_LENGTH + 1)},
            {"title": "Buy screws", "due_at": date(1900, 1, 1)},
            {"title": "Buy screws", "due_at": datetime(9999, 12, 31, 23, 0, tzinfo=UTC)},
        ],
    )
    async def test_invalid_attributes_are_a_domain_error_and_nothing_is_written(
        self, workspace, arguments
    ):
        before = workspace.repositories.snapshot()
        others = {name: value for name, value in arguments.items() if name != "title"}

        with pytest.raises(InvalidTaskError):
            await workspace.create.execute_by_id(OWNER, arguments["title"], **others)

        workspace.assert_untouched(before)


class TestCreateByReference:
    async def test_reports_the_created_task_with_names_and_local_days(self, berlin_workspace):
        workspace = berlin_workspace

        outcome = await workspace.create.execute_by_reference(
            OWNER,
            "Pick worktop material",
            project="renovation",
            due_at="2026-10-05",
            priority="high",
            notes="Oak or laminate",
        )

        assert workspace.data(outcome, "ok") == {
            "created": {
                "title": "Pick worktop material",
                "project": "Home renovation",
                "status": "open",
                "priority": "high",
                "due_at": "2026-10-05",
                "completed_at": None,
                "created_at": "2026-09-18",
            }
        }
        (created,) = [
            task
            for task in await workspace.repositories.tasks.list_by_user(OWNER)
            if task.title == "Pick worktop material"
        ]
        assert created.project_id == workspace.renovation.project_id
        assert created.due_at == datetime(2026, 10, 4, 22, 0, tzinfo=UTC), "local midnight"
        assert created.notes == "Oak or laminate"
        assert created.created_at == CLOCK_NOW
        assert workspace.repositories.write_log == [("save_task", "Pick worktop material")]

    async def test_without_a_project_the_task_is_unfiled_with_normal_priority(self, workspace):
        outcome = await workspace.create.execute_by_reference(OWNER, "Descale the kettle")

        created = workspace.data(outcome, "ok")["created"]
        assert (created["project"], created["priority"], created["due_at"]) == (
            None, "normal", None
        )

    async def test_a_project_name_shared_with_another_owner_is_not_ambiguous(self, workspace):
        outcome = await workspace.create.execute_by_reference(
            OWNER, "Draft the proposal", project="Client work"
        )

        assert workspace.data(outcome, "ok")["created"]["project"] == "Client work"
        (created,) = [
            task
            for task in await workspace.repositories.tasks.list_by_user(OWNER)
            if task.title == "Draft the proposal"
        ]
        assert created.project_id == workspace.clients.project_id

    async def test_an_unmatched_project_is_not_found_and_lists_the_owners_projects(
        self, workspace
    ):
        before = workspace.repositories.snapshot()

        outcome = await workspace.create.execute_by_reference(
            OWNER, "Calibrate the scale", project="Secret lab"
        )

        data = workspace.data(outcome, "not_found")
        assert data["reference"] == "Secret lab"
        assert data["reference_kind"] == "project"
        assert data["available_projects"] == OWNER_PROJECT_NAMES
        assert set(data) == {"reference", "reference_kind", "available_projects", "note"}
        workspace.assert_untouched(before)

    async def test_an_ambiguous_project_files_nothing(self, workspace):
        before = workspace.repositories.snapshot()

        outcome = await workspace.create.execute_by_reference(OWNER, "Buy screws", project="Home")

        data = workspace.data(outcome, "ambiguous_source")
        assert data["reference"] == "Home"
        assert data["reference_kind"] == "project"
        assert [match["title_or_name"] for match in data["matched"]] == [
            "Home office", "Home renovation"
        ]
        workspace.assert_untouched(before)

    @pytest.mark.parametrize(
        "arguments, error_code",
        [
            ({"priority": "urgent"}, "unknown_value"),
            ({"due_at": "next Friday"}, "malformed_date"),
            ({"due_at": "2026-02-30"}, "malformed_date"),
            ({"due_at": "1900-01-01"}, "date_out_of_range"),
            ({"title": "   "}, "invalid_task"),
            ({"title": "x" * (MAX_TASK_TITLE_LENGTH + 1)}, "invalid_task"),
            ({"notes": "x" * (MAX_TASK_NOTES_LENGTH + 1)}, "invalid_task"),
        ],
    )
    async def test_bad_arguments_are_a_filter_error_and_nothing_is_written(
        self, workspace, arguments, error_code
    ):
        before = workspace.repositories.snapshot()
        title = arguments.get("title", "Buy screws")
        others = {name: value for name, value in arguments.items() if name != "title"}

        outcome = await workspace.create.execute_by_reference(OWNER, title, **others)

        data = workspace.data(outcome, "filter_error")
        assert data["error_code"] == error_code
        assert data["message"]
        assert set(data) == {"error_code", "message"}
        workspace.assert_untouched(before)


class TestUpdateById:
    async def test_applies_the_edits_and_stamps_the_update(self, workspace):
        task = await workspace.update.execute_by_id(
            OWNER,
            workspace.tiles.task_id,
            TaskChanges(title="Order bathroom tiles", priority=TaskPriority.HIGH),
        )

        assert (task.title, task.priority) == ("Order bathroom tiles", TaskPriority.HIGH)
        assert task.updated_at == CLOCK_NOW
        assert task.created_at == workspace.tiles.created_at
        assert await workspace.stored(workspace.tiles) == task
        assert workspace.repositories.write_log == [("save_task", "Order bathroom tiles")]

    async def test_completing_sets_the_completion_instant_and_reopening_clears_it(
        self, workspace
    ):
        task_id = workspace.tiles.task_id

        done = await workspace.update.execute_by_id(
            OWNER, task_id, TaskChanges(status=TaskStatus.DONE)
        )
        assert (done.status, done.completed_at) == (TaskStatus.DONE, CLOCK_NOW)

        reopened = await workspace.update.execute_by_id(
            OWNER, task_id, TaskChanges(status=TaskStatus.OPEN)
        )
        assert (reopened.status, reopened.completed_at) == (TaskStatus.OPEN, None)
        assert (await workspace.stored(workspace.tiles)).completed_at is None

    async def test_cancelling_a_finished_task_clears_its_completion_instant(self, workspace):
        task = await workspace.update.execute_by_id(
            OWNER, workspace.hallway.task_id, TaskChanges(status=TaskStatus.CANCELLED)
        )

        assert (task.status, task.completed_at) == (TaskStatus.CANCELLED, None)

    async def test_none_is_a_value_that_clears_the_optional_fields(self, workspace):
        task = await workspace.update.execute_by_id(
            OWNER, workspace.tiles.task_id, TaskChanges(due_at=None, project_id=None, notes=None)
        )

        assert (task.due_at, task.project_id, task.notes) == (None, None, None)
        assert await workspace.stored(workspace.tiles) == task

    async def test_a_date_only_due_day_is_local_midnight_and_a_moment_is_kept(
        self, berlin_workspace
    ):
        workspace = berlin_workspace
        by_day = await workspace.update.execute_by_id(
            OWNER, workspace.tiles.task_id, TaskChanges(due_at=date(2026, 10, 5))
        )
        by_moment = await workspace.update.execute_by_id(
            OWNER,
            workspace.dentist.task_id,
            TaskChanges(due_at=datetime(2026, 10, 5, 16, 30, tzinfo=UTC)),
        )

        assert by_day.due_at == datetime(2026, 10, 4, 22, 0, tzinfo=UTC)
        assert by_moment.due_at == datetime(2026, 10, 5, 16, 30, tzinfo=UTC)

    async def test_a_task_can_be_moved_to_another_project_of_the_owner(self, workspace):
        task = await workspace.update.execute_by_id(
            OWNER, workspace.tiles.task_id, TaskChanges(project_id=workspace.office.project_id)
        )

        assert task.project_id == workspace.office.project_id

    @pytest.mark.parametrize(
        "changes",
        [
            TaskChanges(),
            TaskChanges(title="  Order kitchen tiles ", priority=TaskPriority.NORMAL),
            TaskChanges(status=TaskStatus.OPEN, notes="Matte white"),
            TaskChanges(due_at=datetime(2026, 9, 25, 9, 30, tzinfo=UTC)),
        ],
    )
    async def test_an_edit_that_changes_nothing_writes_nothing(self, workspace, changes):
        before = workspace.repositories.snapshot()

        task = await workspace.update.execute_by_id(OWNER, workspace.tiles.task_id, changes)

        assert task == workspace.tiles
        assert task.updated_at == workspace.tiles.updated_at
        workspace.assert_untouched(before)

    async def test_completing_a_finished_task_again_keeps_its_first_completion(self, workspace):
        before = workspace.repositories.snapshot()

        task = await workspace.update.execute_by_id(
            OWNER, workspace.hallway.task_id, TaskChanges(status=TaskStatus.DONE)
        )

        assert task.completed_at == at(2026, 9, 10)
        workspace.assert_untouched(before)

    async def test_an_unknown_task_id_is_not_found(self, workspace):
        with pytest.raises(TaskNotFoundError):
            await workspace.update.execute_by_id(OWNER, uuid4(), TaskChanges(title="Anything"))

    async def test_another_owners_task_is_not_found_and_stays_as_it_was(self, workspace):
        before = workspace.repositories.snapshot()

        with pytest.raises(TaskNotFoundError):
            await workspace.update.execute_by_id(
                OWNER, workspace.their_laser.task_id, TaskChanges(title="Taken over")
            )

        workspace.assert_untouched(before)

    async def test_a_repository_that_leaks_a_foreign_row_is_caught_by_the_ownership_check(self):
        workspace = await _workspace(_LeakyRepositories())
        before = workspace.repositories.snapshot()

        with pytest.raises(TaskAccessDeniedError):
            await workspace.update.execute_by_id(
                OWNER, workspace.their_laser.task_id, TaskChanges(title="Taken over")
            )

        workspace.assert_untouched(before)

    @pytest.mark.parametrize("which", ["unknown", "foreign"])
    async def test_a_project_the_owner_does_not_have_is_not_found(self, workspace, which):
        before = workspace.repositories.snapshot()
        project_id = uuid4() if which == "unknown" else workspace.their_clients.project_id

        with pytest.raises(ProjectNotFoundError):
            await workspace.update.execute_by_id(
                OWNER, workspace.tiles.task_id, TaskChanges(title="Moved", project_id=project_id)
            )

        workspace.assert_untouched(before)

    async def test_an_invalid_value_is_a_domain_error_and_nothing_is_written(self, workspace):
        before = workspace.repositories.snapshot()

        with pytest.raises(InvalidTaskError):
            await workspace.update.execute_by_id(
                OWNER, workspace.tiles.task_id, TaskChanges(priority=TaskPriority.HIGH, title=" ")
            )

        workspace.assert_untouched(before)


class TestUpdateByReference:
    async def test_an_ambiguous_reference_leaves_the_repository_untouched(self, workspace):
        before = workspace.repositories.snapshot()

        outcome = await workspace.update.execute_by_reference(
            OWNER, "report", None, TaskChangeRequest(set_status="done", set_priority="low")
        )

        data = workspace.data(outcome, "ambiguous_source")
        assert data["reference"] == "report"
        assert data["reference_kind"] == "task"
        assert data["matched"] == [
            {
                "title_or_name": "Quarterly report",
                "project": "Client work",
                "status": "open",
                "due_at": "2026-09-30",
            },
            {
                "title_or_name": "Expense report",
                "project": "Client work",
                "status": "open",
                "due_at": "2026-09-14",
            },
            {
                "title_or_name": "Weekly status report",
                "project": "Client work",
                "status": "open",
                "due_at": None,
            },
        ]
        assert "3 tasks" in data["note"] and "Nothing was changed" in data["note"]
        assert set(data) == {"reference", "reference_kind", "matched", "note"}
        workspace.assert_untouched(before)

    async def test_an_exact_title_among_partial_matches_is_not_ambiguous(self, workspace):
        outcome = await workspace.update.execute_by_reference(
            OWNER, "quarterly REPORT", None, TaskChangeRequest(set_priority="low")
        )

        assert workspace.data(outcome, "ok")["updated"]["title"] == "Quarterly report"
        assert (await workspace.stored(workspace.quarterly)).priority is TaskPriority.LOW
        assert await workspace.stored(workspace.expense) == workspace.expense
        assert await workspace.stored(workspace.weekly) == workspace.weekly

    async def test_a_project_scope_narrows_a_title_collision(self, workspace):
        before = workspace.repositories.snapshot()
        change = TaskChangeRequest(set_priority="high")

        unscoped = await workspace.update.execute_by_reference(OWNER, "Buy paint", None, change)

        assert len(workspace.data(unscoped, "ambiguous_source")["matched"]) == 2
        workspace.assert_untouched(before)

        scoped = await workspace.update.execute_by_reference(OWNER, "Buy paint", "office", change)

        assert workspace.data(scoped, "ok")["updated"]["project"] == "Home office"
        assert (await workspace.stored(workspace.paint_in_office)).priority is TaskPriority.HIGH
        assert await workspace.stored(workspace.paint_in_renovation) == (
            workspace.paint_in_renovation
        )

    async def test_an_ambiguous_project_scope_changes_nothing(self, workspace):
        before = workspace.repositories.snapshot()

        outcome = await workspace.update.execute_by_reference(
            OWNER, "Buy paint", "Home", TaskChangeRequest(set_priority="high")
        )

        data = workspace.data(outcome, "ambiguous_source")
        assert data["reference_kind"] == "project"
        assert [match["title_or_name"] for match in data["matched"]] == [
            "Home office", "Home renovation"
        ]
        workspace.assert_untouched(before)

    async def test_an_unmatched_project_scope_changes_nothing(self, workspace):
        before = workspace.repositories.snapshot()

        outcome = await workspace.update.execute_by_reference(
            OWNER, "Buy paint", "Garage", TaskChangeRequest(set_priority="high")
        )

        data = workspace.data(outcome, "not_found")
        assert data["reference_kind"] == "project"
        assert data["available_projects"] == OWNER_PROJECT_NAMES
        workspace.assert_untouched(before)

    async def test_a_task_outside_the_scope_is_not_found(self, workspace):
        before = workspace.repositories.snapshot()

        outcome = await workspace.update.execute_by_reference(
            OWNER, "Quarterly report", "Home office", TaskChangeRequest(set_priority="low")
        )

        data = workspace.data(outcome, "not_found")
        assert data["reference_kind"] == "task"
        assert "Home office" in data["note"]
        workspace.assert_untouched(before)

    async def test_a_no_op_update_reports_an_empty_change_list_and_writes_nothing(
        self, workspace
    ):
        before = workspace.repositories.snapshot()

        outcome = await workspace.update.execute_by_reference(
            OWNER, "Order kitchen tiles", None, TaskChangeRequest(set_priority="normal")
        )

        data = workspace.data(outcome, "ok")
        assert data["changed"] == []
        assert data["updated"]["title"] == "Order kitchen tiles"
        assert set(data) == {"updated", "changed"}
        workspace.assert_untouched(before)

    @pytest.mark.parametrize(
        "request_",
        [
            TaskChangeRequest(set_title="  Order kitchen tiles  "),
            TaskChangeRequest(set_status="open"),
            TaskChangeRequest(set_due_at="2026-09-25"),
            TaskChangeRequest(set_project="Home renovation"),
            TaskChangeRequest(set_project="renov"),
            TaskChangeRequest(set_notes=" Matte white "),
            TaskChangeRequest(
                set_title="Order kitchen tiles",
                set_status="OPEN",
                set_due_at="2026-09-25",
                set_priority="Normal",
                set_project="home renovation",
                set_notes="Matte white",
            ),
        ],
    )
    async def test_restating_the_current_values_is_a_no_op(self, workspace, request_):
        before = workspace.repositories.snapshot()

        outcome = await workspace.update.execute_by_reference(
            OWNER, "Order kitchen tiles", None, request_
        )

        assert workspace.data(outcome, "ok")["changed"] == []
        workspace.assert_untouched(before)

    async def test_the_same_due_day_keeps_the_stored_time_of_day(self, workspace):
        await workspace.update.execute_by_reference(
            OWNER, "Order kitchen tiles", None, TaskChangeRequest(set_due_at="2026-09-25")
        )

        assert (await workspace.stored(workspace.tiles)).due_at == at(2026, 9, 25, 9, 30)

    async def test_changed_lists_exactly_the_modified_fields(self, workspace):
        outcome = await workspace.update.execute_by_reference(
            OWNER,
            "Order kitchen tiles",
            None,
            TaskChangeRequest(
                set_title="Order bathroom tiles", set_priority="high", set_status="open"
            ),
        )

        data = workspace.data(outcome, "ok")
        assert data["changed"] == [
            {"field": "title", "from": "Order kitchen tiles", "to": "Order bathroom tiles"},
            {"field": "priority", "from": "normal", "to": "high"},
        ]
        assert data["updated"] == {
            "title": "Order bathroom tiles",
            "project": "Home renovation",
            "status": "open",
            "priority": "high",
            "due_at": "2026-09-25",
            "completed_at": None,
            "created_at": "2026-09-08",
        }
        stored = await workspace.stored(workspace.tiles)
        assert (stored.title, stored.priority) == ("Order bathroom tiles", TaskPriority.HIGH)
        assert stored.updated_at == CLOCK_NOW
        assert workspace.repositories.write_log == [("save_task", "Order bathroom tiles")]

    async def test_only_the_addressed_task_is_written(self, workspace):
        before_projects, before_tasks = workspace.repositories.snapshot()

        await workspace.update.execute_by_reference(
            OWNER, "Order kitchen tiles", None, TaskChangeRequest(set_priority="high")
        )

        after_projects, after_tasks = workspace.repositories.snapshot()
        changed_ids = [key for key in before_tasks if before_tasks[key] != after_tasks[key]]
        assert changed_ids == [workspace.tiles.task_id]
        assert after_projects == before_projects and set(after_tasks) == set(before_tasks)

    async def test_set_status_done_sets_the_completion_instant(self, workspace):
        outcome = await workspace.update.execute_by_reference(
            OWNER, "Order kitchen tiles", None, TaskChangeRequest(set_status="done")
        )

        data = workspace.data(outcome, "ok")
        assert data["changed"] == [
            {"field": "status", "from": "open", "to": "done"},
            {"field": "completed_at", "from": None, "to": "2026-09-18"},
        ]
        stored = await workspace.stored(workspace.tiles)
        assert (stored.status, stored.completed_at) == (TaskStatus.DONE, CLOCK_NOW)

    async def test_reopening_clears_the_completion_instant(self, workspace):
        outcome = await workspace.update.execute_by_reference(
            OWNER, "Paint hallway ceiling", None, TaskChangeRequest(set_status="open")
        )

        data = workspace.data(outcome, "ok")
        assert data["changed"] == [
            {"field": "status", "from": "done", "to": "open"},
            {"field": "completed_at", "from": "2026-09-10", "to": None},
        ]
        stored = await workspace.stored(workspace.hallway)
        assert (stored.status, stored.completed_at) == (TaskStatus.OPEN, None)

    async def test_cancelling_a_finished_task_clears_the_completion_instant(self, workspace):
        outcome = await workspace.update.execute_by_reference(
            OWNER, "Paint hallway ceiling", None, TaskChangeRequest(set_status="cancelled")
        )

        assert [change["field"] for change in workspace.data(outcome, "ok")["changed"]] == [
            "status", "completed_at"
        ]
        stored = await workspace.stored(workspace.hallway)
        assert (stored.status, stored.completed_at) == (TaskStatus.CANCELLED, None)

    @pytest.mark.parametrize("clear_word", ["none", "None", "  NONE "])
    async def test_set_due_at_none_removes_the_due_date(self, workspace, clear_word):
        outcome = await workspace.update.execute_by_reference(
            OWNER, "Order kitchen tiles", None, TaskChangeRequest(set_due_at=clear_word)
        )

        assert workspace.data(outcome, "ok")["changed"] == [
            {"field": "due_at", "from": "2026-09-25", "to": None}
        ]
        assert (await workspace.stored(workspace.tiles)).due_at is None

    async def test_a_new_due_day_is_stored_as_its_local_midnight(self, berlin_workspace):
        workspace = berlin_workspace

        outcome = await workspace.update.execute_by_reference(
            OWNER, "Book dentist appointment", None, TaskChangeRequest(set_due_at="2026-10-05")
        )

        assert workspace.data(outcome, "ok")["changed"] == [
            {"field": "due_at", "from": None, "to": "2026-10-05"}
        ]
        assert (await workspace.stored(workspace.dentist)).due_at == datetime(
            2026, 10, 4, 22, 0, tzinfo=UTC
        )

    async def test_set_project_none_unfiles_the_task(self, workspace):
        outcome = await workspace.update.execute_by_reference(
            OWNER, "Order kitchen tiles", None, TaskChangeRequest(set_project="none")
        )

        assert workspace.data(outcome, "ok")["changed"] == [
            {"field": "project", "from": "Home renovation", "to": None}
        ]
        assert (await workspace.stored(workspace.tiles)).project_id is None

    async def test_set_project_moves_the_task_to_the_owners_project_of_that_name(
        self, workspace
    ):
        outcome = await workspace.update.execute_by_reference(
            OWNER, "Order kitchen tiles", None, TaskChangeRequest(set_project="Client work")
        )

        assert workspace.data(outcome, "ok")["changed"] == [
            {"field": "project", "from": "Home renovation", "to": "Client work"}
        ]
        stored = await workspace.stored(workspace.tiles)
        assert stored.project_id == workspace.clients.project_id
        assert stored.project_id != workspace.their_clients.project_id

    async def test_an_unmatched_target_project_changes_nothing(self, workspace):
        before = workspace.repositories.snapshot()

        outcome = await workspace.update.execute_by_reference(
            OWNER,
            "Order kitchen tiles",
            None,
            TaskChangeRequest(set_project="Secret lab", set_priority="high"),
        )

        data = workspace.data(outcome, "not_found")
        assert data["reference"] == "Secret lab"
        assert data["reference_kind"] == "project"
        assert data["available_projects"] == OWNER_PROJECT_NAMES
        workspace.assert_untouched(before)

    async def test_an_ambiguous_target_project_applies_none_of_the_edits(self, workspace):
        before = workspace.repositories.snapshot()

        outcome = await workspace.update.execute_by_reference(
            OWNER,
            "Book dentist appointment",
            None,
            TaskChangeRequest(set_project="Home", set_priority="high", set_status="done"),
        )

        data = workspace.data(outcome, "ambiguous_source")
        assert data["reference"] == "Home"
        assert data["reference_kind"] == "project"
        workspace.assert_untouched(before)

    async def test_notes_can_be_rewritten_and_blank_text_clears_them(self, workspace):
        rewritten = await workspace.update.execute_by_reference(
            OWNER, "dentist", None, TaskChangeRequest(set_notes="Bring the insurance card")
        )
        cleared = await workspace.update.execute_by_reference(
            OWNER, "dentist", None, TaskChangeRequest(set_notes="  ")
        )

        assert workspace.data(rewritten, "ok")["changed"] == [
            {
                "field": "notes",
                "from": "Ask about the night guard",
                "to": "Bring the insurance card",
            }
        ]
        assert workspace.data(cleared, "ok")["changed"] == [
            {"field": "notes", "from": "Bring the insurance card", "to": None}
        ]
        assert (await workspace.stored(workspace.dentist)).notes is None

    async def test_a_task_can_be_addressed_by_its_notes(self, workspace):
        outcome = await workspace.update.execute_by_reference(
            OWNER, "night guard", None, TaskChangeRequest(set_priority="high")
        )

        assert workspace.data(outcome, "ok")["updated"]["title"] == "Book dentist appointment"

    @pytest.mark.parametrize("reference", ["Hire a plumber", "", "   "])
    async def test_an_unmatched_task_is_not_found(self, workspace, reference):
        before = workspace.repositories.snapshot()

        outcome = await workspace.update.execute_by_reference(
            OWNER, reference, None, TaskChangeRequest(set_status="done")
        )

        data = workspace.data(outcome, "not_found")
        assert data["reference"] == reference
        assert data["reference_kind"] == "task"
        assert set(data) == {"reference", "reference_kind", "note"}
        workspace.assert_untouched(before)

    @pytest.mark.parametrize("reference", ["Order kitchen tiles", "report", "Hire a plumber"])
    async def test_an_update_without_any_set_argument_is_a_filter_error(
        self, workspace, reference
    ):
        before = workspace.repositories.snapshot()

        outcome = await workspace.update.execute_by_reference(
            OWNER, reference, None, TaskChangeRequest()
        )

        data = workspace.data(outcome, "filter_error")
        assert data["error_code"] == "no_changes_requested"
        assert "set_status" in data["message"] and "set_notes" in data["message"]
        workspace.assert_untouched(before)

    @pytest.mark.parametrize(
        "request_, error_code",
        [
            (TaskChangeRequest(set_status="paused"), "unknown_value"),
            (TaskChangeRequest(set_priority="urgent"), "unknown_value"),
            (TaskChangeRequest(set_due_at="tomorrow"), "malformed_date"),
            (TaskChangeRequest(set_due_at="2101-01-01"), "date_out_of_range"),
            (TaskChangeRequest(set_title="   ", set_priority="high"), "invalid_task"),
            (
                TaskChangeRequest(set_priority="high", set_notes="x" * (MAX_TASK_NOTES_LENGTH + 1)),
                "invalid_task",
            ),
        ],
    )
    async def test_a_bad_value_stops_the_whole_update(self, workspace, request_, error_code):
        before = workspace.repositories.snapshot()

        outcome = await workspace.update.execute_by_reference(
            OWNER, "Order kitchen tiles", None, request_
        )

        assert workspace.data(outcome, "filter_error")["error_code"] == error_code
        workspace.assert_untouched(before)

    async def test_another_owners_identically_titled_task_is_never_matched_or_touched(
        self, workspace
    ):
        outcome = await workspace.update.execute_by_reference(
            OWNER, "Book dentist appointment", None, TaskChangeRequest(set_status="done")
        )

        assert workspace.data(outcome, "ok")["updated"]["status"] == "done"
        assert (await workspace.stored(workspace.dentist)).status is TaskStatus.DONE
        assert await workspace.stored(workspace.their_dentist, STRANGER) == workspace.their_dentist
        assert workspace.repositories.write_log == [("save_task", "Book dentist appointment")]

    async def test_a_title_only_another_owner_has_is_not_found(self, workspace):
        before = workspace.repositories.snapshot()

        outcome = await workspace.update.execute_by_reference(
            OWNER, "Calibrate the laser", None, TaskChangeRequest(set_status="done")
        )

        workspace.data(outcome, "not_found")
        workspace.assert_untouched(before)

    async def test_a_leaked_foreign_row_is_refused_by_the_ownership_check(self):
        workspace = await _workspace(_LeakyRepositories())
        before = workspace.repositories.snapshot()

        with pytest.raises(TaskAccessDeniedError):
            await workspace.update.execute_by_reference(
                OWNER, "Calibrate the laser", None, TaskChangeRequest(set_status="done")
            )

        workspace.assert_untouched(before)


class TestDeleteById:
    async def test_removes_the_task_and_nothing_else(self, workspace):
        _projects, before_tasks = workspace.repositories.snapshot()

        assert await workspace.delete.execute_by_id(OWNER, workspace.tiles.task_id) is None

        _projects, after_tasks = workspace.repositories.snapshot()
        assert set(before_tasks) - set(after_tasks) == {workspace.tiles.task_id}
        assert all(after_tasks[key] == before_tasks[key] for key in after_tasks)
        assert workspace.repositories.write_log == [("delete_task", "Order kitchen tiles")]

    async def test_an_unknown_task_id_is_not_found(self, workspace):
        before = workspace.repositories.snapshot()

        with pytest.raises(TaskNotFoundError):
            await workspace.delete.execute_by_id(OWNER, uuid4())

        workspace.assert_untouched(before)

    async def test_another_owners_task_is_not_found_and_survives(self, workspace):
        before = workspace.repositories.snapshot()

        with pytest.raises(TaskNotFoundError):
            await workspace.delete.execute_by_id(OWNER, workspace.their_laser.task_id)

        workspace.assert_untouched(before)

    async def test_a_repository_that_leaks_a_foreign_row_is_caught_by_the_ownership_check(self):
        workspace = await _workspace(_LeakyRepositories())
        before = workspace.repositories.snapshot()

        with pytest.raises(TaskAccessDeniedError):
            await workspace.delete.execute_by_id(OWNER, workspace.their_laser.task_id)

        workspace.assert_untouched(before)


class TestDeleteByReference:
    async def test_an_ambiguous_reference_leaves_the_repository_untouched(self, workspace):
        before = workspace.repositories.snapshot()

        outcome = await workspace.delete.execute_by_reference(OWNER, "report")

        data = workspace.data(outcome, "ambiguous_source")
        assert data["reference_kind"] == "task"
        assert [match["title_or_name"] for match in data["matched"]] == [
            "Quarterly report", "Expense report", "Weekly status report"
        ]
        assert set(data["matched"][0]) == {"title_or_name", "project", "status", "due_at"}
        workspace.assert_untouched(before)

    async def test_two_tasks_that_look_interchangeable_are_still_refused(self, workspace):
        await workspace.repositories.given(
            a_task(OWNER, "Water the plants", created_at=at(2026, 9, 11)),
            a_task(OWNER, "Water the plants", created_at=at(2026, 9, 11)),
        )
        before = workspace.repositories.snapshot()

        outcome = await workspace.delete.execute_by_reference(OWNER, "Water the plants")

        assert len(workspace.data(outcome, "ambiguous_source")["matched"]) == 2
        workspace.assert_untouched(before)

    async def test_deletes_the_single_match_and_reports_it(self, workspace):
        _projects, before_tasks = workspace.repositories.snapshot()

        outcome = await workspace.delete.execute_by_reference(OWNER, "kitchen tiles")

        assert workspace.data(outcome, "ok") == {
            "deleted": {
                "title": "Order kitchen tiles",
                "project": "Home renovation",
                "status": "open",
                "priority": "normal",
                "due_at": "2026-09-25",
                "completed_at": None,
                "created_at": "2026-09-08",
            }
        }
        _projects, after_tasks = workspace.repositories.snapshot()
        assert set(before_tasks) - set(after_tasks) == {workspace.tiles.task_id}
        assert all(after_tasks[key] == before_tasks[key] for key in after_tasks)
        assert workspace.repositories.write_log == [("delete_task", "Order kitchen tiles")]

    async def test_an_exact_title_among_partial_matches_is_deleted_alone(self, workspace):
        outcome = await workspace.delete.execute_by_reference(OWNER, "Expense report")

        assert workspace.data(outcome, "ok")["deleted"]["title"] == "Expense report"
        assert await workspace.stored(workspace.expense) is None
        assert await workspace.stored(workspace.quarterly) == workspace.quarterly
        assert await workspace.stored(workspace.weekly) == workspace.weekly

    async def test_a_project_scope_narrows_a_title_collision(self, workspace):
        before = workspace.repositories.snapshot()

        unscoped = await workspace.delete.execute_by_reference(OWNER, "Buy paint")

        workspace.data(unscoped, "ambiguous_source")
        workspace.assert_untouched(before)

        scoped = await workspace.delete.execute_by_reference(OWNER, "Buy paint", "renovation")

        assert workspace.data(scoped, "ok")["deleted"]["project"] == "Home renovation"
        assert await workspace.stored(workspace.paint_in_renovation) is None
        assert await workspace.stored(workspace.paint_in_office) == workspace.paint_in_office

    async def test_an_ambiguous_project_scope_deletes_nothing(self, workspace):
        before = workspace.repositories.snapshot()

        outcome = await workspace.delete.execute_by_reference(OWNER, "Buy paint", "Home")

        assert workspace.data(outcome, "ambiguous_source")["reference_kind"] == "project"
        workspace.assert_untouched(before)

    async def test_an_unmatched_project_scope_deletes_nothing(self, workspace):
        before = workspace.repositories.snapshot()

        outcome = await workspace.delete.execute_by_reference(OWNER, "Buy paint", "Garage")

        data = workspace.data(outcome, "not_found")
        assert data["reference_kind"] == "project"
        assert data["available_projects"] == OWNER_PROJECT_NAMES
        workspace.assert_untouched(before)

    @pytest.mark.parametrize("reference", ["Hire a plumber", "", "  "])
    async def test_an_unmatched_task_is_not_found(self, workspace, reference):
        before = workspace.repositories.snapshot()

        outcome = await workspace.delete.execute_by_reference(OWNER, reference)

        data = workspace.data(outcome, "not_found")
        assert data["reference"] == reference
        assert data["reference_kind"] == "task"
        assert set(data) == {"reference", "reference_kind", "note"}
        workspace.assert_untouched(before)

    async def test_finished_tasks_can_be_deleted_too(self, workspace):
        outcome = await workspace.delete.execute_by_reference(OWNER, "Paint hallway ceiling")

        deleted = workspace.data(outcome, "ok")["deleted"]
        assert (deleted["status"], deleted["completed_at"]) == ("done", "2026-09-10")
        assert await workspace.stored(workspace.hallway) is None

    async def test_another_owners_identically_titled_task_is_never_matched_or_touched(
        self, workspace
    ):
        outcome = await workspace.delete.execute_by_reference(OWNER, "Book dentist appointment")

        workspace.data(outcome, "ok")
        assert await workspace.stored(workspace.dentist) is None
        assert await workspace.stored(workspace.their_dentist, STRANGER) == workspace.their_dentist
        assert workspace.repositories.write_log == [("delete_task", "Book dentist appointment")]

    async def test_a_title_only_another_owner_has_is_not_found(self, workspace):
        before = workspace.repositories.snapshot()

        outcome = await workspace.delete.execute_by_reference(OWNER, "Calibrate the laser")

        workspace.data(outcome, "not_found")
        workspace.assert_untouched(before)

    async def test_a_leaked_foreign_row_is_refused_by_the_ownership_check(self):
        workspace = await _workspace(_LeakyRepositories())
        before = workspace.repositories.snapshot()

        with pytest.raises(TaskAccessDeniedError):
            await workspace.delete.execute_by_reference(OWNER, "Calibrate the laser")

        workspace.assert_untouched(before)
