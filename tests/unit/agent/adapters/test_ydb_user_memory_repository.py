"""The memory vault repository: mapper, owner scoping, literal search, migration."""
from datetime import datetime, timezone

import pytest
import ydb

from src.agent.adapters.persistence.ydb_user_memory_repository import (
    UserMemoryMapper,
    YDBUserMemoryRepository,
    escape_like_substring,
)
from src.agent.domain.entities.user_memory import UserMemory
from src.agent.ports.user_memory_repository import UserMemoryRepository
from src.shared.domain.value_objects.user_id import UserId
from tests.unit.agent.adapters.ydb_fakes import (
    FakePool,
    declared_columns,
    load_agent_migration,
    mapper_columns,
    one_line,
)

MIGRATION_VERSION = "20260901000200"
VIEW_CLAUSE = "FROM user_memory VIEW idx_user_memory_user_id WHERE user_id = $user_id"


class _Row:
    """An SDK-style row: columns are attributes."""

    def __init__(self, memory: UserMemory):
        self.memory_id = memory.memory_id
        self.user_id = str(memory.user_id)
        self.content = memory.content
        self.topic = memory.topic
        self.created_at = memory.created_at.replace(tzinfo=None)
        self.updated_at = memory.updated_at.replace(tzinfo=None)


def _memory(topic="cycling") -> UserMemory:
    return UserMemory(
        memory_id=UserMemory.generate_id(),
        user_id=UserId.generate(),
        content="Prefers to plan long rides on Saturdays",
        topic=topic,
        created_at=datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 9, 3, 18, 30, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("topic", ["cycling", None])
def test_mapper_round_trips_an_entry_including_a_missing_topic(topic):
    mapper = UserMemoryMapper()
    original = _memory(topic=topic)

    assert mapper.to_domain(_Row(original)) == original

    params = mapper.to_ydb_params(original)
    assert params["$user_id"] == str(original.user_id)
    assert params["$topic"] == topic
    assert set(params) == set(mapper.get_column_params())


def test_repository_implements_the_port():
    assert isinstance(YDBUserMemoryRepository(FakePool()), UserMemoryRepository)


@pytest.mark.parametrize(
    "raw, escaped",
    [
        ("100%_sure", "100!%!_sure"),
        ("wow!", "wow!!"),
        ("plain text", "plain text"),
        ("!%_", "!!!%!_"),
    ],
)
def test_like_wildcards_and_the_escape_character_itself_are_escaped(raw, escaped):
    assert escape_like_substring(raw) == escaped


async def test_save_upserts_every_column():
    pool = FakePool()
    memory = _memory(topic=None)

    saved = await YDBUserMemoryRepository(pool).save(memory)

    assert saved is memory
    assert (
        "UPSERT INTO user_memory (content, created_at, memory_id, topic, updated_at, user_id)"
        in pool.last_query
    )
    assert "$pattern" not in pool.last_query
    assert pool.last_parameters["$topic"].value is None
    assert pool.last_parameters["$memory_id"].value_type == ydb.PrimitiveType.Utf8


async def test_find_by_user_id_names_the_owner_index_and_lists_freshest_first():
    memory = _memory()
    pool = FakePool(rows=[_Row(memory)])

    found = await YDBUserMemoryRepository(pool).find_by_user_id(memory.user_id, limit=30)

    assert found == [memory]
    query = one_line(pool.last_query)
    assert VIEW_CLAUSE in query
    assert query.endswith("ORDER BY updated_at DESC LIMIT $limit;")
    assert pool.last_parameters["$limit"].value == 30
    assert pool.last_parameters["$user_id"].value == str(memory.user_id)


async def test_search_escapes_wildcards_and_is_scoped_by_the_owner_index():
    pool = FakePool()
    user_id = UserId.generate()

    await YDBUserMemoryRepository(pool).search_by_user_id(user_id, query="100%_sure", limit=5)

    query = one_line(pool.last_query)
    assert VIEW_CLAUSE in query
    assert pool.last_parameters["$user_id"].value == str(user_id)
    # The wildcards arrive escaped, so the input is matched as a literal substring.
    assert pool.last_parameters["$pattern"].value == "%100!%!_sure%"
    assert pool.last_parameters["$pattern"].value_type == ydb.PrimitiveType.Utf8
    assert pool.last_parameters["$limit"].value == 5
    assert query.count("ESCAPE '!'") == 2


async def test_search_is_case_insensitive_without_casts_and_tolerates_a_missing_topic():
    pool = FakePool()

    await YDBUserMemoryRepository(pool).search_by_user_id(
        UserId.generate(), query="Rides", limit=5
    )

    query = one_line(pool.last_query)
    assert "Unicode::ToLower(content) LIKE Unicode::ToLower($pattern)" in query
    assert "topic IS NOT NULL AND Unicode::ToLower(topic) LIKE Unicode::ToLower($pattern)" in query
    assert "CAST(" not in query
    assert query.endswith("ORDER BY updated_at DESC LIMIT $limit;")


async def test_delete_requires_both_the_memory_id_and_the_owner():
    pool = FakePool()
    user_id = UserId.generate()

    await YDBUserMemoryRepository(pool).delete("memory-1", user_id)

    query = one_line(pool.last_query)
    assert "DELETE FROM user_memory WHERE memory_id = $memory_id AND user_id = $user_id;" in query
    assert pool.last_parameters["$memory_id"].value == "memory-1"
    assert pool.last_parameters["$user_id"].value == str(user_id)


async def test_delete_all_selects_the_owner_rows_through_the_index():
    pool = FakePool()
    user_id = UserId.generate()

    await YDBUserMemoryRepository(pool).delete_all_by_user_id(user_id)

    query = one_line(pool.last_query)
    assert f"DELETE FROM user_memory ON SELECT memory_id {VIEW_CLAUSE};" in query
    assert pool.last_parameters["$user_id"].value == str(user_id)


async def test_count_is_scoped_by_the_owner_index():
    pool = FakePool(rows=[{"total": 12}])
    user_id = UserId.generate()

    count = await YDBUserMemoryRepository(pool).count_by_user_id(user_id)

    assert count == 12
    query = one_line(pool.last_query)
    assert f"SELECT COUNT(*) AS total {VIEW_CLAUSE};" in query
    assert pool.last_parameters["$user_id"].value == str(user_id)


async def test_count_without_a_result_row_is_zero():
    assert await YDBUserMemoryRepository(FakePool()).count_by_user_id(UserId.generate()) == 0


async def test_find_oldest_orders_by_created_at_ascending():
    memory = _memory()
    pool = FakePool(rows=[_Row(memory)])

    oldest = await YDBUserMemoryRepository(pool).find_oldest_by_user_id(memory.user_id)

    assert oldest == memory
    query = one_line(pool.last_query)
    assert VIEW_CLAUSE in query
    assert query.endswith("ORDER BY created_at ASC LIMIT 1;")


async def test_find_oldest_in_an_empty_vault_is_none():
    assert await YDBUserMemoryRepository(FakePool()).find_oldest_by_user_id(UserId.generate()) is None


class TestUserMemoryMigration:
    def test_declares_the_table_its_index_and_six_typed_columns(self):
        migration = load_agent_migration(MIGRATION_VERSION)
        artifacts = migration.get_artifacts()

        assert artifacts["tables"] == ["user_memory"]
        assert artifacts["indexes"] == [("user_memory", "idx_user_memory_user_id")]
        assert len(artifacts["columns"]) == 6
        assert declared_columns(migration, "user_memory") == {
            "memory_id": "Utf8",
            "user_id": "Utf8",
            "content": "Utf8",
            "topic": "Utf8?",
            "created_at": "Timestamp",
            "updated_at": "Timestamp",
        }

    def test_declared_columns_match_what_the_mapper_writes(self):
        declared = declared_columns(load_agent_migration(MIGRATION_VERSION), "user_memory")

        assert mapper_columns(UserMemoryMapper()) == declared

    async def test_creates_the_table_idempotently_with_its_inline_index(self):
        pool = FakePool()

        await load_agent_migration(MIGRATION_VERSION).up(pool)

        (statement,) = [one_line(query) for query, _parameters in pool.calls]
        assert statement.startswith("CREATE TABLE IF NOT EXISTS user_memory (")
        assert "PRIMARY KEY (memory_id)" in statement
        assert "INDEX idx_user_memory_user_id GLOBAL ON (user_id)" in statement
        assert "content Utf8 NOT NULL" in statement
        assert "topic Utf8?," in statement

    async def test_failed_statement_is_reported_and_re_raised(self, capsys):
        pool = FakePool(error=ydb.issues.Unavailable("node is down"))

        with pytest.raises(ydb.issues.Unavailable):
            await load_agent_migration(MIGRATION_VERSION).up(pool)

        assert "node is down" in capsys.readouterr().err
