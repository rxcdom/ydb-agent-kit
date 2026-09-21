"""The projects repository: mapper, the owner index, owner scoping, migration."""
from __future__ import annotations

from datetime import timezone
from uuid import uuid4

import pytest
import ydb

from src.shared.domain.exceptions import PersistenceError
from src.shared.domain.value_objects.user_id import UserId
from src.shared.infrastructure.database.ydb.types import format_ydb_type
from src.tasks.adapters.persistence.ydb_project_repository import (
    INDEX_USER_ID,
    TABLE_NAME,
    ProjectMapper,
    YDBProjectRepository,
)
from src.tasks.domain.entities.project import Project
from src.tasks.ports.project_repository import ProjectRepository
from tests.unit.tasks.builders import a_project, at
from tests.unit.tasks.ydb_fakes import (
    FakePool,
    FakeTx,
    assert_ddl_declares,
    declared_columns,
    load_tasks_migration,
    mapper_columns,
)

MIGRATION_VERSION = "20260901000400"
OWNER = UserId.generate()
COLUMNS = "project_id, user_id, name, description, created_at, updated_at"


class _Row:
    """An SDK-style row: columns are attributes and timestamps come back naive, in UTC."""

    def __init__(self, project: Project):
        self.project_id = str(project.project_id)
        self.user_id = str(project.user_id)
        self.name = project.name
        self.description = project.description
        self.created_at = project.created_at.replace(tzinfo=None)
        self.updated_at = project.updated_at.replace(tzinfo=None)


def _project(description="Kitchen and bathroom remodel") -> Project:
    return a_project(OWNER, "Home renovation", description, created_at=at(2026, 1, 12, 8, 30))


class TestMapper:
    @pytest.mark.parametrize("description", ["Kitchen and bathroom remodel", None])
    def test_round_trips_a_project_through_a_row_with_naive_timestamps(self, description):
        mapper = ProjectMapper()
        original = _project(description)

        restored = mapper.to_domain(_Row(original))

        assert restored == original
        assert restored.description == description
        assert restored.created_at.tzinfo == timezone.utc

    def test_a_plain_mapping_is_accepted_as_a_row(self):
        original = _project()

        assert ProjectMapper().to_domain(vars(_Row(original))) == original

    def test_a_row_without_a_required_value_is_refused(self):
        row = _Row(_project())
        row.name = None

        with pytest.raises(ValueError, match="name"):
            ProjectMapper().to_domain(row)

    def test_parameters_cover_every_column_with_plain_values(self):
        mapper = ProjectMapper()
        project = _project(description=None)

        params = mapper.to_ydb_params(project)

        assert params == {
            "$project_id": str(project.project_id),
            "$user_id": str(OWNER),
            "$name": "Home renovation",
            "$description": None,
            "$created_at": project.created_at,
            "$updated_at": project.updated_at,
        }
        assert set(params) == set(mapper.get_column_params())

    def test_only_the_description_is_declared_optional(self):
        assert mapper_columns(ProjectMapper()) == {
            "project_id": "Utf8",
            "user_id": "Utf8",
            "name": "Utf8",
            "description": "Utf8?",
            "created_at": "Timestamp",
            "updated_at": "Timestamp",
        }


def test_repository_implements_the_port():
    assert isinstance(YDBProjectRepository(FakePool()), ProjectRepository)


async def test_save_upserts_every_column_with_typed_parameters():
    pool = FakePool()
    project = _project(description=None)

    await YDBProjectRepository(pool).save(project)

    assert (
        "UPSERT INTO projects (created_at, description, name, project_id, updated_at, user_id) "
        "VALUES ($created_at, $description, $name, $project_id, $updated_at, $user_id);"
    ) in pool.last_query
    assert "DECLARE $description AS Utf8?;" in pool.last_query
    parameters = pool.last_parameters
    assert parameters["$user_id"].value == str(OWNER)
    assert parameters["$name"].value_type == ydb.PrimitiveType.Utf8
    assert parameters["$description"].value is None
    assert format_ydb_type(parameters["$description"].value_type) == "Utf8?"
    assert parameters["$created_at"].value_type == ydb.PrimitiveType.Timestamp


async def test_save_inside_a_transaction_runs_on_its_context_without_committing():
    pool, tx = FakePool(), FakeTx()

    await YDBProjectRepository(pool).save(_project(), tx)

    assert pool.calls == []
    ((statement, parameters, commit),) = tx.calls
    assert "UPSERT INTO projects" in statement
    assert parameters["$name"].value == "Home renovation"
    assert commit is False


async def test_find_by_id_reads_the_primary_key_guarded_by_the_owner():
    project = _project()
    pool = FakePool(rows=[_Row(project)])

    found = await YDBProjectRepository(pool).find_by_id(OWNER, project.project_id)

    assert found == project
    assert pool.last_query.endswith(
        f"SELECT {COLUMNS} FROM projects WHERE project_id = $project_id AND user_id = $user_id;"
    )
    assert "VIEW" not in pool.last_query
    assert pool.last_values() == {"$project_id": str(project.project_id), "$user_id": str(OWNER)}


async def test_find_by_id_without_a_row_is_none():
    assert await YDBProjectRepository(FakePool()).find_by_id(OWNER, uuid4()) is None


async def test_list_by_user_names_the_owner_index_orders_by_name_and_sets_no_limit():
    projects = [_project(), a_project(OWNER, "Garden", created_at=at(2026, 2, 1))]
    pool = FakePool(rows=[_Row(project) for project in projects])

    listed = await YDBProjectRepository(pool).list_by_user(OWNER)

    assert listed == projects
    assert pool.last_query.endswith(
        f"SELECT {COLUMNS} FROM projects VIEW idx_projects_user_id WHERE user_id = $user_id "
        "ORDER BY name, project_id;"
    )
    assert "LIMIT" not in pool.last_query
    assert pool.last_values() == {"$user_id": str(OWNER)}


async def test_every_read_is_scoped_to_the_owner():
    pool = FakePool()
    repository = YDBProjectRepository(pool)

    await repository.find_by_id(OWNER, uuid4())
    await repository.list_by_user(OWNER)

    for query, parameters in pool.calls:
        assert "user_id = $user_id" in query
        assert parameters["$user_id"].value == str(OWNER)


async def test_the_table_name_can_be_replaced():
    pool = FakePool()

    await YDBProjectRepository(pool, "scratch_projects").list_by_user(OWNER)

    assert "FROM scratch_projects VIEW idx_projects_user_id" in pool.last_query


async def test_an_sdk_failure_surfaces_as_a_persistence_error():
    pool = FakePool(error=ydb.issues.Unavailable("node is down"))

    with pytest.raises(PersistenceError, match="projects"):
        await YDBProjectRepository(pool).list_by_user(OWNER)


class TestProjectsMigration:
    def test_declares_the_table_its_index_and_six_typed_columns(self):
        migration = load_tasks_migration(MIGRATION_VERSION)
        artifacts = migration.get_artifacts()

        assert (migration.version, migration.description) == (
            MIGRATION_VERSION, "Create projects table"
        )
        assert artifacts["tables"] == ["projects"]
        assert artifacts["indexes"] == [("projects", "idx_projects_user_id")]
        assert len(artifacts["columns"]) == 6
        assert declared_columns(migration, "projects") == {
            "project_id": "Utf8",
            "user_id": "Utf8",
            "name": "Utf8",
            "description": "Utf8?",
            "created_at": "Timestamp",
            "updated_at": "Timestamp",
        }

    def test_declared_columns_match_what_the_mapper_writes(self):
        declared = declared_columns(load_tasks_migration(MIGRATION_VERSION), "projects")

        assert mapper_columns(ProjectMapper()) == declared

    def test_the_repository_names_exactly_the_table_and_index_the_migration_creates(self):
        artifacts = load_tasks_migration(MIGRATION_VERSION).get_artifacts()

        assert [TABLE_NAME] == artifacts["tables"]
        assert [("projects", INDEX_USER_ID)] == artifacts["indexes"]

    async def test_creates_the_table_idempotently_with_its_inline_index(self):
        pool = FakePool()
        migration = load_tasks_migration(MIGRATION_VERSION)

        await migration.up(pool)

        assert len(pool.calls) == 1
        statement = pool.last_query
        assert statement.startswith("CREATE TABLE IF NOT EXISTS projects (")
        assert "PRIMARY KEY (project_id)" in statement
        assert "INDEX idx_projects_user_id GLOBAL ON (user_id)" in statement
        assert statement.count(" INDEX ") == 1
        assert_ddl_declares(statement, declared_columns(migration, "projects"))

    async def test_a_failed_statement_is_reported_and_re_raised(self, capsys):
        pool = FakePool(error=ydb.issues.Unavailable("node is down"))

        with pytest.raises(ydb.issues.Unavailable):
            await load_tasks_migration(MIGRATION_VERSION).up(pool)

        assert "node is down" in capsys.readouterr().err
