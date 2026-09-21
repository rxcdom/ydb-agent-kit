"""The tasks repository: mapper, the index every query names, owner scoping, migration."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
import ydb

from src.shared.domain.exceptions import PersistenceError
from src.shared.domain.value_objects.user_id import UserId
from src.shared.infrastructure.database.ydb.types import format_ydb_type
from src.tasks.adapters.persistence.ydb_task_repository import (
    INDEX_PROJECT_ID,
    INDEX_USER_CREATED,
    INDEX_USER_DUE,
    TABLE_NAME,
    TaskMapper,
    YDBTaskRepository,
)
from src.tasks.domain.entities.task import Task
from src.tasks.domain.exceptions import InvalidTaskError
from src.tasks.domain.value_objects.date_axis import DateAxis
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus
from src.tasks.ports.task_repository import ProjectActivity, TaskRepository, TaskSearchCriteria
from tests.unit.tasks.builders import a_task, at
from tests.unit.tasks.ydb_fakes import (
    FakePool,
    FakeTx,
    assert_ddl_declares,
    declared_columns,
    load_tasks_migration,
    mapper_columns,
)

MIGRATION_VERSION = "20260901000500"
OWNER = UserId.generate()
OWNER_CONDITION = "WHERE user_id = $user_id"

COLUMNS = (
    "task_id, user_id, project_id, title, notes, status, priority, "
    "due_at, completed_at, created_at, updated_at"
)

# Which index each date axis reads through. completed_at has none of its own:
# the owner prefix of the creation index keeps the read inside the owner's rows.
AXIS_INDEXES = [
    (DateAxis.CREATED, "created_at", "idx_tasks_user_created"),
    (DateAxis.DUE, "due_at", "idx_tasks_user_due"),
    (DateAxis.COMPLETED, "completed_at", "idx_tasks_user_created"),
]


class _Row:
    """An SDK-style row: columns are attributes and timestamps come back naive, in UTC."""

    def __init__(self, task: Task):
        self.task_id = str(task.task_id)
        self.user_id = str(task.user_id)
        self.project_id = None if task.project_id is None else str(task.project_id)
        self.title = task.title
        self.notes = task.notes
        self.status = task.status.value
        self.priority = task.priority.value
        self.due_at = _naive(task.due_at)
        self.completed_at = _naive(task.completed_at)
        self.created_at = _naive(task.created_at)
        self.updated_at = _naive(task.updated_at)


def _naive(instant):
    return None if instant is None else instant.replace(tzinfo=None)


def _full_task() -> Task:
    return a_task(
        OWNER,
        "Paint hallway ceiling",
        project_id=uuid4(),
        status=TaskStatus.DONE,
        priority=TaskPriority.HIGH,
        created_at=at(2026, 9, 1, 8, 15),
        due_at=at(2026, 9, 4, 0),
        completed_at=at(2026, 9, 5, 17, 40),
        notes="Two coats of matte white",
    )


def _bare_task() -> Task:
    return a_task(OWNER, "Book dentist appointment", created_at=at(2026, 9, 12, 9, 5))


class TestMapper:
    @pytest.mark.parametrize("build", [_full_task, _bare_task], ids=["every-field", "nullables"])
    def test_round_trips_a_task_through_a_row_with_naive_timestamps(self, build):
        mapper = TaskMapper()
        original = build()

        restored = mapper.to_domain(_Row(original))

        assert restored == original
        assert restored.created_at.tzinfo == timezone.utc
        assert restored.updated_at.tzinfo == timezone.utc

    def test_nullable_columns_come_back_as_none(self):
        restored = TaskMapper().to_domain(_Row(_bare_task()))

        assert (restored.project_id, restored.notes) == (None, None)
        assert (restored.due_at, restored.completed_at) == (None, None)

    def test_a_plain_mapping_is_accepted_as_a_row(self):
        original = _full_task()

        assert TaskMapper().to_domain(vars(_Row(original))) == original

    def test_a_row_without_a_required_value_is_refused(self):
        row = _Row(_bare_task())
        row.title = None

        with pytest.raises(ValueError, match="title"):
            TaskMapper().to_domain(row)

    def test_parameters_cover_every_column_with_plain_values(self):
        mapper = TaskMapper()
        task = _full_task()

        params = mapper.to_ydb_params(task)

        assert set(params) == set(mapper.get_column_params())
        assert params["$task_id"] == str(task.task_id)
        assert params["$user_id"] == str(OWNER)
        assert params["$project_id"] == str(task.project_id)
        assert (params["$status"], params["$priority"]) == ("done", "high")
        assert params["$due_at"] == task.due_at
        assert params["$completed_at"] == task.completed_at

    def test_missing_optional_values_are_written_as_none(self):
        params = TaskMapper().to_ydb_params(_bare_task())

        assert [params[name] for name in ("$project_id", "$notes", "$due_at", "$completed_at")] == (
            [None] * 4
        )

    def test_a_contradictory_status_and_completion_pair_is_never_written(self):
        task = _bare_task()
        task.status = TaskStatus.DONE

        with pytest.raises(InvalidTaskError):
            TaskMapper().to_ydb_params(task)

    def test_only_the_nullable_columns_are_declared_optional(self):
        assert mapper_columns(TaskMapper()) == {
            "task_id": "Utf8",
            "user_id": "Utf8",
            "project_id": "Utf8?",
            "title": "Utf8",
            "notes": "Utf8?",
            "status": "Utf8",
            "priority": "Utf8",
            "due_at": "Timestamp?",
            "completed_at": "Timestamp?",
            "created_at": "Timestamp",
            "updated_at": "Timestamp",
        }


def test_repository_implements_the_port():
    assert isinstance(YDBTaskRepository(FakePool()), TaskRepository)


class TestWritesAndPrimaryKeyReads:
    async def test_save_upserts_every_column_with_typed_parameters(self):
        pool = FakePool()
        task = _bare_task()

        await YDBTaskRepository(pool).save(task)

        assert (
            "UPSERT INTO tasks (completed_at, created_at, due_at, notes, priority, project_id, "
            "status, task_id, title, updated_at, user_id) VALUES ($completed_at, $created_at, "
            "$due_at, $notes, $priority, $project_id, $status, $task_id, $title, $updated_at, "
            "$user_id);"
        ) in pool.last_query
        assert "DECLARE $due_at AS Timestamp?;" in pool.last_query
        assert "DECLARE $title AS Utf8;" in pool.last_query
        parameters = pool.last_parameters
        assert parameters["$user_id"].value == str(OWNER)
        assert parameters["$task_id"].value_type == ydb.PrimitiveType.Utf8
        assert parameters["$created_at"].value == task.created_at
        assert parameters["$created_at"].value_type == ydb.PrimitiveType.Timestamp
        assert parameters["$due_at"].value is None
        assert format_ydb_type(parameters["$due_at"].value_type) == "Timestamp?"
        assert format_ydb_type(parameters["$project_id"].value_type) == "Utf8?"
        assert set(parameters) == set(TaskMapper().get_column_params())

    async def test_save_inside_a_transaction_runs_on_its_context_without_committing(self):
        pool, tx = FakePool(), FakeTx()

        await YDBTaskRepository(pool).save(_full_task(), tx)

        assert pool.calls == []
        ((statement, parameters, commit),) = tx.calls
        assert "UPSERT INTO tasks" in statement
        assert parameters["$status"].value == "done"
        assert commit is False

    async def test_save_refuses_a_contradictory_pair_before_touching_the_database(self):
        pool = FakePool()
        task = _full_task()
        task.completed_at = None

        with pytest.raises(InvalidTaskError):
            await YDBTaskRepository(pool).save(task)

        assert pool.calls == []

    async def test_find_by_id_reads_the_primary_key_guarded_by_the_owner(self):
        task = _full_task()
        pool = FakePool(rows=[_Row(task)])

        found = await YDBTaskRepository(pool).find_by_id(OWNER, task.task_id)

        assert found == task
        assert pool.last_query.endswith(
            f"SELECT {COLUMNS} FROM tasks WHERE task_id = $task_id AND user_id = $user_id;"
        )
        assert "VIEW" not in pool.last_query
        assert pool.last_values() == {"$task_id": str(task.task_id), "$user_id": str(OWNER)}

    async def test_find_by_id_without_a_row_is_none(self):
        assert await YDBTaskRepository(FakePool()).find_by_id(OWNER, uuid4()) is None

    async def test_delete_is_a_hard_delete_that_needs_both_the_id_and_the_owner(self):
        pool = FakePool()
        task_id = uuid4()

        await YDBTaskRepository(pool).delete(OWNER, task_id)

        assert pool.last_query.endswith(
            "DELETE FROM tasks WHERE task_id = $task_id AND user_id = $user_id;"
        )
        assert pool.last_values() == {"$task_id": str(task_id), "$user_id": str(OWNER)}
        assert len(pool.calls) == 1


class TestListings:
    async def test_list_by_user_names_the_owner_creation_index_and_sets_no_limit(self):
        rows = [_full_task(), _bare_task()]
        pool = FakePool(rows=[_Row(task) for task in rows])

        listed = await YDBTaskRepository(pool).list_by_user(OWNER)

        assert listed == rows
        assert pool.last_query.endswith(
            f"SELECT {COLUMNS} FROM tasks VIEW idx_tasks_user_created {OWNER_CONDITION} "
            "ORDER BY created_at, task_id;"
        )
        assert "LIMIT" not in pool.last_query
        assert pool.last_values() == {"$user_id": str(OWNER)}

    async def test_list_by_project_names_the_project_index_and_keeps_the_owner_guard(self):
        pool = FakePool()
        project_id = uuid4()

        await YDBTaskRepository(pool).list_by_project(OWNER, project_id)

        assert pool.last_query.endswith(
            f"SELECT {COLUMNS} FROM tasks VIEW idx_tasks_project_id "
            "WHERE project_id = $project_id AND user_id = $user_id ORDER BY created_at, task_id;"
        )
        assert "LIMIT" not in pool.last_query
        assert pool.last_values() == {"$project_id": str(project_id), "$user_id": str(OWNER)}


class TestSearch:
    @pytest.mark.parametrize("axis, column, index", AXIS_INDEXES)
    async def test_search_names_the_index_of_its_axis(self, axis, column, index):
        pool = FakePool(rows=[_Row(_full_task())])
        window_start, window_end = at(2026, 9, 1, 0), at(2026, 10, 1, 0)
        criteria = TaskSearchCriteria(axis=axis, window_start=window_start, window_end=window_end)

        found = await YDBTaskRepository(pool).search(OWNER, criteria)

        assert [task.title for task in found] == ["Paint hallway ceiling"]
        assert (
            f"SELECT {COLUMNS} FROM tasks VIEW {index} {OWNER_CONDITION} "
            f"AND {column} IS NOT NULL AND {column} >= $window_start AND {column} < $window_end "
            f"ORDER BY {column} ASC, task_id ASC"
        ) in pool.last_query
        assert pool.last_values() == {
            "$user_id": str(OWNER), "$window_start": window_start, "$window_end": window_end
        }
        assert pool.last_parameters["$window_start"].value_type == ydb.PrimitiveType.Timestamp

    async def test_search_without_a_limit_renders_no_limit_clause(self):
        pool = FakePool()

        await YDBTaskRepository(pool).search(OWNER, TaskSearchCriteria())

        assert "LIMIT" not in pool.last_query and "OFFSET" not in pool.last_query
        assert "$limit" not in pool.last_query
        assert pool.last_values() == {"$user_id": str(OWNER)}

    async def test_search_pages_newest_first_when_asked(self):
        pool = FakePool()
        criteria = TaskSearchCriteria(axis=DateAxis.DUE)

        await YDBTaskRepository(pool).search(OWNER, criteria, limit=20, offset=40, descending=True)

        assert pool.last_query.endswith(
            "ORDER BY due_at DESC, task_id DESC LIMIT $limit OFFSET $offset;"
        )
        assert "DECLARE $limit AS Uint64;" in pool.last_query
        assert pool.last_parameters["$limit"].value == 20
        assert pool.last_parameters["$offset"].value == 40
        assert pool.last_parameters["$limit"].value_type == ydb.PrimitiveType.Uint64

    async def test_an_offset_without_a_limit_is_a_programming_error(self):
        pool = FakePool()

        with pytest.raises(ValueError, match="offset needs a limit"):
            await YDBTaskRepository(pool).search(OWNER, TaskSearchCriteria(), offset=10)

        assert pool.calls == []

    async def test_an_open_window_side_renders_no_condition(self):
        pool = FakePool()
        only_end = TaskSearchCriteria(axis=DateAxis.DUE, window_end=at(2026, 9, 18, 0))

        await YDBTaskRepository(pool).search(OWNER, only_end)

        assert "due_at < $window_end" in pool.last_query
        assert "$window_start" not in pool.last_query
        assert set(pool.last_values()) == {"$user_id", "$window_end"}

    async def test_every_narrowing_filter_becomes_a_typed_condition(self):
        pool = FakePool()
        project_id = uuid4()
        criteria = TaskSearchCriteria(
            axis=DateAxis.DUE,
            project_id=project_id,
            statuses=frozenset({TaskStatus.OPEN, TaskStatus.DONE}),
            priorities=frozenset({TaskPriority.HIGH}),
            text="Tiles",
        )

        await YDBTaskRepository(pool).search(OWNER, criteria)

        assert (
            f"FROM tasks VIEW idx_tasks_user_due {OWNER_CONDITION} AND due_at IS NOT NULL "
            "AND project_id = $project_id AND status IN ($status_0, $status_1) "
            "AND priority IN ($priority_0) "
            'AND (Unicode::ToLower(title) LIKE $text_pattern ESCAPE "!" '
            'OR Unicode::ToLower(notes) LIKE $text_pattern ESCAPE "!")'
        ) in pool.last_query
        assert pool.last_values() == {
            "$user_id": str(OWNER),
            "$project_id": str(project_id),
            "$status_0": "done",
            "$status_1": "open",
            "$priority_0": "high",
            "$text_pattern": "%tiles%",
        }
        assert "CAST(" not in pool.last_query
        assert pool.last_parameters["$status_0"].value_type == ydb.PrimitiveType.Utf8

    @pytest.mark.parametrize(
        "text, pattern",
        [
            ("100%_Sure", "%100!%!_sure%"),
            ("Wow!", "%wow!!%"),
            ("plain text", "%plain text%"),
            ("!%_", "%!!!%!_%"),
        ],
    )
    async def test_the_text_is_matched_as_a_literal_lower_cased_substring(self, text, pattern):
        pool = FakePool()

        await YDBTaskRepository(pool).search(OWNER, TaskSearchCriteria(text=text))

        assert pool.last_parameters["$text_pattern"].value == pattern

    async def test_absent_filters_render_no_conditions(self):
        pool = FakePool()

        await YDBTaskRepository(pool).search(OWNER, TaskSearchCriteria())

        assert f"{OWNER_CONDITION} AND created_at IS NOT NULL ORDER BY" in pool.last_query
        for fragment in ("project_id =", "status IN", "priority IN", "LIKE"):
            assert fragment not in pool.last_query


class TestCounts:
    @pytest.mark.parametrize("axis, column, index", AXIS_INDEXES)
    async def test_count_names_the_index_of_its_axis(self, axis, column, index):
        pool = FakePool(rows=[{"total": 7}])
        criteria = TaskSearchCriteria(
            axis=axis, window_start=at(2026, 9, 1, 0), statuses=frozenset({TaskStatus.OPEN})
        )

        total = await YDBTaskRepository(pool).count(OWNER, criteria)

        assert total == 7
        assert pool.last_query.endswith(
            f"SELECT COUNT(*) AS total FROM tasks VIEW {index} {OWNER_CONDITION} "
            f"AND {column} IS NOT NULL AND {column} >= $window_start AND status IN ($status_0);"
        )
        assert pool.last_values() == {
            "$user_id": str(OWNER), "$window_start": at(2026, 9, 1, 0), "$status_0": "open"
        }

    async def test_count_without_a_result_row_is_zero(self):
        assert await YDBTaskRepository(FakePool()).count(OWNER, TaskSearchCriteria()) == 0

    @pytest.mark.parametrize("axis, column, index", AXIS_INDEXES)
    async def test_counting_tasks_without_a_date_ignores_the_window(self, axis, column, index):
        pool = FakePool(rows=[{"total": 4}])
        project_id = uuid4()
        criteria = TaskSearchCriteria(
            axis=axis,
            window_start=at(2026, 9, 1, 0),
            window_end=at(2026, 10, 1, 0),
            project_id=project_id,
        )

        total = await YDBTaskRepository(pool).count_without_axis_date(OWNER, criteria)

        assert total == 4
        assert pool.last_query.endswith(
            f"SELECT COUNT(*) AS total FROM tasks VIEW {index} {OWNER_CONDITION} "
            f"AND {column} IS NULL AND project_id = $project_id;"
        )
        assert pool.last_values() == {"$user_id": str(OWNER), "$project_id": str(project_id)}


class TestAggregates:
    @pytest.mark.parametrize("axis, column, index", AXIS_INDEXES)
    async def test_date_bounds_names_the_index_of_the_axis(self, axis, column, index):
        first, last = datetime(2026, 1, 5, 8, 0), datetime(2026, 9, 17, 22, 30)
        pool = FakePool(rows=[{"first_value": first, "last_value": last}])

        bounds = await YDBTaskRepository(pool).date_bounds(OWNER, axis)

        assert bounds == (at(2026, 1, 5, 8), at(2026, 9, 17, 22, 30)), "naive values are UTC"
        assert pool.last_query.endswith(
            f"SELECT MIN({column}) AS first_value, MAX({column}) AS last_value "
            f"FROM tasks VIEW {index} {OWNER_CONDITION};"
        )
        assert pool.last_values() == {"$user_id": str(OWNER)}

    @pytest.mark.parametrize("rows", [[], [{"first_value": None, "last_value": None}]])
    async def test_an_axis_without_any_value_has_no_bounds(self, rows):
        assert await YDBTaskRepository(FakePool(rows=rows)).date_bounds(OWNER, DateAxis.DUE) is None

    async def test_summarise_by_project_groups_the_owners_rows_through_the_creation_index(self):
        project_id = uuid4()
        pool = FakePool(
            rows=[
                {
                    "project_id": str(project_id),
                    "status": "open",
                    "task_count": 3,
                    "first_created_at": datetime(2026, 8, 1, 9, 0),
                    "last_updated_at": datetime(2026, 9, 2, 10, 0),
                },
                {
                    "project_id": None,
                    "status": "done",
                    "task_count": 1,
                    "first_created_at": datetime(2026, 7, 1, 9, 0),
                    "last_updated_at": datetime(2026, 7, 4, 18, 0),
                },
            ]
        )

        activity = await YDBTaskRepository(pool).summarise_by_project(OWNER)

        assert activity == [
            ProjectActivity(project_id, TaskStatus.OPEN, 3, at(2026, 8, 1, 9), at(2026, 9, 2, 10)),
            ProjectActivity(None, TaskStatus.DONE, 1, at(2026, 7, 1, 9), at(2026, 7, 4, 18)),
        ]
        assert pool.last_query.endswith(
            "SELECT project_id, status, COUNT(*) AS task_count, "
            "MIN(created_at) AS first_created_at, MAX(updated_at) AS last_updated_at "
            f"FROM tasks VIEW idx_tasks_user_created {OWNER_CONDITION} "
            "GROUP BY project_id, status;"
        )
        assert pool.last_values() == {"$user_id": str(OWNER)}

    async def test_an_owner_without_tasks_has_no_activity(self):
        assert await YDBTaskRepository(FakePool()).summarise_by_project(OWNER) == []


class TestEveryStatement:
    async def _run_every_method(self, repository: YDBTaskRepository) -> None:
        criteria = TaskSearchCriteria(axis=DateAxis.DUE, text="tiles")
        await repository.find_by_id(OWNER, uuid4())
        await repository.delete(OWNER, uuid4())
        await repository.list_by_user(OWNER)
        await repository.list_by_project(OWNER, uuid4())
        await repository.search(OWNER, criteria)
        await repository.count(OWNER, criteria)
        await repository.count_without_axis_date(OWNER, criteria)
        for axis in DateAxis:
            await repository.date_bounds(OWNER, axis)
        await repository.summarise_by_project(OWNER)

    async def test_every_read_and_delete_is_scoped_to_the_owner(self):
        pool = FakePool()

        await self._run_every_method(YDBTaskRepository(pool))

        assert len(pool.calls) == 11
        for query, parameters in pool.calls:
            assert "user_id = $user_id" in query, query
            assert "DECLARE $user_id AS Utf8;" in query, query
            assert parameters["$user_id"].value == str(OWNER)

    async def test_every_statement_that_is_not_a_primary_key_access_names_an_index(self):
        pool = FakePool()

        await self._run_every_method(YDBTaskRepository(pool))

        by_key, through_an_index = pool.calls[:2], pool.calls[2:]
        assert all(" VIEW " not in query for query, _parameters in by_key)
        assert all("task_id = $task_id" in query for query, _parameters in by_key)
        assert all(" VIEW idx_tasks_" in query for query, _parameters in through_an_index)

    async def test_the_table_name_can_be_replaced(self):
        pool = FakePool()

        await YDBTaskRepository(pool, "scratch_tasks").list_by_user(OWNER)

        assert "FROM scratch_tasks VIEW idx_tasks_user_created" in pool.last_query

    async def test_an_sdk_failure_surfaces_as_a_persistence_error(self):
        pool = FakePool(error=ydb.issues.Unavailable("node is down"))

        with pytest.raises(PersistenceError, match="tasks"):
            await YDBTaskRepository(pool).list_by_user(OWNER)


class TestTasksMigration:
    def test_declares_the_table_three_indexes_and_eleven_typed_columns(self):
        migration = load_tasks_migration(MIGRATION_VERSION)
        artifacts = migration.get_artifacts()

        assert (migration.version, migration.description) == (
            MIGRATION_VERSION, "Create tasks table"
        )
        assert artifacts["tables"] == ["tasks"]
        assert artifacts["indexes"] == [
            ("tasks", "idx_tasks_user_created"),
            ("tasks", "idx_tasks_user_due"),
            ("tasks", "idx_tasks_project_id"),
        ]
        assert len(artifacts["columns"]) == 11
        assert declared_columns(migration, "tasks") == {
            "task_id": "Utf8",
            "user_id": "Utf8",
            "project_id": "Utf8?",
            "title": "Utf8",
            "notes": "Utf8?",
            "status": "Utf8",
            "priority": "Utf8",
            "due_at": "Timestamp?",
            "completed_at": "Timestamp?",
            "created_at": "Timestamp",
            "updated_at": "Timestamp",
        }

    def test_declared_columns_match_what_the_mapper_writes(self):
        declared = declared_columns(load_tasks_migration(MIGRATION_VERSION), "tasks")

        assert mapper_columns(TaskMapper()) == declared

    def test_the_repository_names_exactly_the_table_and_indexes_the_migration_creates(self):
        artifacts = load_tasks_migration(MIGRATION_VERSION).get_artifacts()

        assert [TABLE_NAME] == artifacts["tables"]
        assert {INDEX_USER_CREATED, INDEX_USER_DUE, INDEX_PROJECT_ID} == {
            index for _table, index in artifacts["indexes"]
        }

    async def test_creates_the_table_idempotently_with_its_inline_indexes(self):
        pool = FakePool()
        migration = load_tasks_migration(MIGRATION_VERSION)

        await migration.up(pool)

        assert len(pool.calls) == 1
        statement = pool.last_query
        assert statement.startswith("CREATE TABLE IF NOT EXISTS tasks (")
        assert "PRIMARY KEY (task_id)" in statement
        assert "INDEX idx_tasks_user_created GLOBAL ON (user_id, created_at)" in statement
        assert "INDEX idx_tasks_user_due GLOBAL ON (user_id, due_at)" in statement
        assert "INDEX idx_tasks_project_id GLOBAL ON (project_id)" in statement
        assert statement.count(" INDEX ") == 3
        assert_ddl_declares(statement, declared_columns(migration, "tasks"))

    async def test_a_failed_statement_is_reported_and_re_raised(self, capsys):
        pool = FakePool(error=ydb.issues.Unavailable("node is down"))

        with pytest.raises(ydb.issues.Unavailable):
            await load_tasks_migration(MIGRATION_VERSION).up(pool)

        assert "node is down" in capsys.readouterr().err
