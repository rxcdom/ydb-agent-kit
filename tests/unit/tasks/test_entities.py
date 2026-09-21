from datetime import datetime, timedelta, timezone

import pytest

from src.shared.domain.exceptions import AccessDeniedError, NotFoundError, ValidationError
from src.shared.domain.value_objects.user_id import UserId
from src.tasks.domain.entities.project import (
    MAX_PROJECT_DESCRIPTION_LENGTH,
    MAX_PROJECT_NAME_LENGTH,
    Project,
)
from src.tasks.domain.entities.task import MAX_TASK_NOTES_LENGTH, MAX_TASK_TITLE_LENGTH, Task
from src.tasks.domain.exceptions import (
    InvalidDateWindowError,
    InvalidProjectError,
    InvalidTaskError,
    ProjectAccessDeniedError,
    ProjectNotFoundError,
    TaskAccessDeniedError,
    TaskNotFoundError,
)
from src.tasks.domain.value_objects.date_axis import DateAxis
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus
from tests.unit.tasks.builders import NOW, UTC, a_project, a_task, at

OWNER = UserId.generate()


def _assert_invariant(task: Task) -> None:
    assert (task.status is TaskStatus.DONE) == (task.completed_at is not None)


def test_a_new_task_is_open_with_no_completion_instant():
    task = Task.create(OWNER, "  Order kitchen tiles  ", NOW)

    assert task.status is TaskStatus.OPEN
    assert task.completed_at is None
    assert task.priority is TaskPriority.NORMAL
    assert task.title == "Order kitchen tiles"
    assert task.created_at == task.updated_at == NOW
    _assert_invariant(task)


def test_done_without_a_completion_instant_cannot_be_constructed():
    fields = {**vars(a_task(OWNER, "Paint")), "status": TaskStatus.DONE}

    with pytest.raises(InvalidTaskError, match="must carry its completion instant"):
        Task(**fields)


@pytest.mark.parametrize("status", [TaskStatus.OPEN, TaskStatus.CANCELLED])
def test_a_completion_instant_without_done_cannot_be_constructed(status):
    fields = {**vars(a_task(OWNER, "Paint")), "status": status, "completed_at": NOW}

    with pytest.raises(InvalidTaskError, match="only a done task"):
        Task(**fields)


def test_complete_sets_the_pair_and_is_idempotent():
    task = a_task(OWNER, "Paint")
    finished_at = NOW + timedelta(days=1)

    task.complete(finished_at)
    assert (task.status, task.completed_at) == (TaskStatus.DONE, finished_at)

    task.complete(finished_at + timedelta(days=5))
    assert task.completed_at == finished_at, "the first completion instant is kept"
    _assert_invariant(task)


def test_reopen_and_cancel_clear_the_completion_instant():
    for leave_done in (Task.reopen, Task.cancel):
        task = a_task(OWNER, "Paint", status=TaskStatus.DONE, completed_at=NOW)

        leave_done(task)

        assert task.completed_at is None
        assert task.status is not TaskStatus.DONE
        _assert_invariant(task)


def test_every_status_change_keeps_the_invariant():
    statuses = list(TaskStatus)
    for start in statuses:
        for target in statuses:
            task = a_task(OWNER, "Paint", status=start)

            task.change_status(target, NOW + timedelta(hours=1))

            assert task.status is target
            _assert_invariant(task)


def test_check_invariants_catches_a_pair_broken_by_direct_assignment():
    task = a_task(OWNER, "Paint")
    task.status = TaskStatus.DONE

    with pytest.raises(InvalidTaskError):
        task.check_invariants()


def test_a_completion_instant_must_be_aware():
    with pytest.raises(InvalidTaskError, match="timezone-aware"):
        a_task(OWNER, "Paint").complete(datetime(2026, 9, 18, 12, 0))


def test_instants_are_normalised_to_utc():
    plus_three = timezone(timedelta(hours=3))
    task = a_task(OWNER, "Paint", created_at=datetime(2026, 9, 18, 15, 0, tzinfo=plus_three))

    assert task.created_at == at(2026, 9, 18, 12)
    assert task.created_at.tzinfo == UTC


def test_naive_instants_are_rejected():
    with pytest.raises(InvalidTaskError, match="created_at must be timezone-aware"):
        a_task(OWNER, "Paint", created_at=datetime(2026, 9, 18, 12, 0))
    with pytest.raises(InvalidTaskError, match="due_at must be timezone-aware"):
        a_task(OWNER, "Paint", due_at=datetime(2026, 9, 18, 12, 0))


@pytest.mark.parametrize("year", [1, 1969, 2101, 9999])
def test_a_due_instant_outside_the_supported_range_is_rejected(year):
    with pytest.raises(InvalidTaskError, match="supported calendar range"):
        a_task(OWNER, "Paint", due_at=datetime(year, 6, 1, tzinfo=UTC))


@pytest.mark.parametrize("title", ["", "   ", "x" * (MAX_TASK_TITLE_LENGTH + 1)])
def test_title_must_have_one_to_two_hundred_characters(title):
    with pytest.raises(InvalidTaskError):
        Task.create(OWNER, title, NOW)
    with pytest.raises(InvalidTaskError):
        a_task(OWNER, "Paint").rename(title)


def test_title_at_the_limit_is_accepted():
    assert len(Task.create(OWNER, "x" * MAX_TASK_TITLE_LENGTH, NOW).title) == MAX_TASK_TITLE_LENGTH


def test_notes_are_trimmed_capped_and_blank_means_none():
    task = a_task(OWNER, "Paint", notes="  two coats  ")
    assert task.notes == "two coats"

    task.rewrite_notes("   ")
    assert task.notes is None

    with pytest.raises(InvalidTaskError):
        task.rewrite_notes("x" * (MAX_TASK_NOTES_LENGTH + 1))


def test_unknown_status_and_priority_are_rejected():
    with pytest.raises(InvalidTaskError, match="unknown task status"):
        Task(**{**vars(a_task(OWNER, "Paint")), "status": "paused"})
    with pytest.raises(InvalidTaskError, match="unknown task priority"):
        a_task(OWNER, "Paint").reprioritise("urgent")


def test_text_values_of_the_vocabularies_are_accepted():
    task = Task(**{**vars(a_task(OWNER, "Paint")), "status": "cancelled", "priority": "high"})

    assert task.status is TaskStatus.CANCELLED
    assert task.priority is TaskPriority.HIGH


def test_field_mutators():
    project = a_project(OWNER, "Garden")
    task = a_task(OWNER, "Paint")

    task.reschedule(at(2026, 10, 1, 0))
    task.move_to_project(project.project_id)
    task.reprioritise(TaskPriority.LOW)
    task.touch(NOW + timedelta(minutes=5))

    assert task.due_at == at(2026, 10, 1, 0)
    assert task.project_id == project.project_id
    assert task.priority is TaskPriority.LOW
    assert task.updated_at == NOW + timedelta(minutes=5)

    task.reschedule(None)
    task.move_to_project(None)
    assert task.due_at is None and task.project_id is None


def test_date_on_returns_the_instant_of_each_axis():
    task = a_task(
        OWNER, "Paint", status=TaskStatus.DONE, due_at=at(2026, 9, 20), completed_at=at(2026, 9, 19)
    )

    assert task.date_on(DateAxis.CREATED) == NOW
    assert task.date_on(DateAxis.DUE) == at(2026, 9, 20)
    assert task.date_on(DateAxis.COMPLETED) == at(2026, 9, 19)
    assert a_task(OWNER, "Paint").date_on(DateAxis.DUE) is None


def test_only_the_completed_axis_is_limited_to_finished_tasks():
    assert DateAxis.COMPLETED.statuses_with_value == frozenset({TaskStatus.DONE})
    assert DateAxis.CREATED.statuses_with_value == frozenset(TaskStatus)
    assert DateAxis.DUE.statuses_with_value == frozenset(TaskStatus)


def test_ownership_checks():
    stranger = UserId.generate()
    task = a_task(OWNER, "Paint")
    project = a_project(OWNER, "Garden")

    task.ensure_owned_by(OWNER)
    project.ensure_owned_by(OWNER)
    with pytest.raises(TaskAccessDeniedError):
        task.ensure_owned_by(stranger)
    with pytest.raises(ProjectAccessDeniedError):
        project.ensure_owned_by(stranger)


def test_project_name_and_description_rules():
    project = Project.create(OWNER, "  Garden  ", "   ", NOW)
    assert project.name == "Garden"
    assert project.description is None

    for name in ("", "  ", "x" * (MAX_PROJECT_NAME_LENGTH + 1)):
        with pytest.raises(InvalidProjectError):
            Project.create(OWNER, name, None, NOW)
    with pytest.raises(InvalidProjectError):
        Project.create(OWNER, "Garden", "x" * (MAX_PROJECT_DESCRIPTION_LENGTH + 1), NOW)
    with pytest.raises(InvalidProjectError, match="timezone-aware"):
        Project.create(OWNER, "Garden", None, datetime(2026, 9, 18))


def test_module_errors_subclass_the_shared_hierarchy():
    assert issubclass(TaskNotFoundError, NotFoundError)
    assert issubclass(ProjectNotFoundError, NotFoundError)
    assert issubclass(TaskAccessDeniedError, AccessDeniedError)
    assert issubclass(ProjectAccessDeniedError, AccessDeniedError)
    for error in (InvalidTaskError, InvalidProjectError, InvalidDateWindowError):
        assert issubclass(error, ValidationError)
