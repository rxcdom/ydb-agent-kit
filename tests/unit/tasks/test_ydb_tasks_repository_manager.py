"""The repository manager of the tasks module, and the migrations the module ships."""
from __future__ import annotations

from datetime import date

import pytest

from src.shared.domain.value_objects.user_id import UserId
from src.shared.infrastructure.database.migration.discovery import discover_migration_files
from src.tasks.adapters.persistence.ydb_tasks_repository_manager import YDBTasksRepositoryManager
from src.tasks.application.seed_demo_workspace import SeedDemoWorkspaceUseCase
from src.tasks.ports.project_repository import ProjectRepository
from src.tasks.ports.repository_manager import TasksRepositoryManager
from src.tasks.ports.task_repository import TaskRepository
from tests.unit.tasks.builders import a_project, a_task
from tests.unit.tasks.ydb_fakes import FakePool, TransactionalFakePool

OWNER = UserId.generate()
ANCHOR = date(2026, 9, 21)


def test_the_manager_exposes_both_repositories_of_the_port():
    manager = YDBTasksRepositoryManager(FakePool())

    assert isinstance(manager, TasksRepositoryManager)
    assert isinstance(manager.projects, ProjectRepository)
    assert isinstance(manager.tasks, TaskRepository)


def test_repositories_are_created_once():
    manager = YDBTasksRepositoryManager(FakePool())

    assert manager.projects is manager.projects
    assert manager.tasks is manager.tasks


async def test_operations_share_one_transaction_context_and_return_in_order():
    pool = TransactionalFakePool()
    manager = YDBTasksRepositoryManager(pool)
    project = a_project(OWNER, "Garden")
    task = a_task(OWNER, "Mow the lawn", project=project)

    results = await manager.execute_in_transaction(
        [
            lambda tx: manager.projects.save(project, tx),
            lambda tx: manager.tasks.save(task, tx),
        ]
    )

    assert results == [None, None]
    assert pool.calls == [], "nothing runs outside the transaction"
    statements = [statement for statement, _parameters, _commit in pool.tx.calls]
    assert "UPSERT INTO projects" in statements[0]
    assert "UPSERT INTO tasks" in statements[1]
    assert all(commit is False for _statement, _parameters, commit in pool.tx.calls)
    assert pool.tx.rolled_back is False


async def test_a_failing_operation_rolls_the_transaction_back():
    pool = TransactionalFakePool()
    manager = YDBTasksRepositoryManager(pool)

    async def failing(_tx):
        raise RuntimeError("the second write failed")

    with pytest.raises(RuntimeError, match="second write failed"):
        await manager.execute_in_transaction(
            [lambda tx: manager.projects.save(a_project(OWNER, "Garden"), tx), failing]
        )

    assert pool.tx.rolled_back is True


async def test_the_demo_seed_goes_through_one_transaction_of_upserts():
    pool = TransactionalFakePool()

    summary = await SeedDemoWorkspaceUseCase(YDBTasksRepositoryManager(pool)).execute(
        OWNER, ANCHOR
    )

    assert pool.calls == []
    statements = [statement for statement, _parameters, _commit in pool.tx.calls]
    assert len(statements) == summary.projects_count + summary.tasks_count
    assert all("UPSERT INTO projects" in statement for statement in statements[:4])
    assert all("UPSERT INTO tasks" in statement for statement in statements[4:])
    owners = {parameters["$user_id"].value for _statement, parameters, _commit in pool.tx.calls}
    assert owners == {str(OWNER)}


def test_the_module_ships_exactly_its_two_migrations_in_order():
    versions = [
        (migration_file.version, migration_file.path.name)
        for migration_file in discover_migration_files()
        if migration_file.module_name == "tasks"
    ]

    assert versions == [
        ("20260901000400", "20260901000400_create_projects_table.py"),
        ("20260901000500", "20260901000500_create_tasks_table.py"),
    ]
