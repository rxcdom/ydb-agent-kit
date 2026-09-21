from datetime import datetime, timezone

import pytest

from src.agent.adapters.persistence.ydb_chat_agent_context_repository import (
    ChatAgentContextMapper,
    YDBChatAgentContextRepository,
)
from src.agent.domain.entities.chat_agent_context import ChatAgentContext
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.ports.chat_agent_context_repository import ChatAgentContextRepository
from src.shared.domain.value_objects.user_id import UserId
from tests.unit.agent.adapters.ydb_fakes import (
    FakePool,
    FakeTx,
    declared_columns,
    load_agent_migration,
    mapper_columns,
    one_line,
)

MIGRATION_VERSION = "20260901000300"


def _context(**helper_values) -> ChatAgentContext:
    return ChatAgentContext(
        chat_id=ChatId.generate(),
        user_id=UserId.generate(),
        created_at=datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 9, 17, 14, 5, tzinfo=timezone.utc),
        **helper_values,
    )


def _recorded_context() -> ChatAgentContext:
    return _context(
        last_tool="query_tasks",
        last_window_from="2026-09-01",
        last_window_to="2026-09-17",
        last_date_field="created",
        last_project="Home renovation",
    )


def _row(context: ChatAgentContext) -> dict:
    params = ChatAgentContextMapper().to_ydb_params(context)
    row = {name.removeprefix("$"): value for name, value in params.items()}
    row["created_at"] = row["created_at"].replace(tzinfo=None)
    row["updated_at"] = row["updated_at"].replace(tzinfo=None)
    return row


@pytest.mark.parametrize("context", [_recorded_context(), _context()], ids=["recorded", "empty"])
def test_mapper_round_trips_a_context(context):
    restored = ChatAgentContextMapper().to_domain(_row(context))

    assert restored == context
    assert restored.has_context() is context.has_context()


def test_mapper_writes_every_column_it_declares():
    mapper = ChatAgentContextMapper()

    assert set(mapper.to_ydb_params(_context())) == set(mapper.get_ydb_type_map())


def test_repository_implements_the_port():
    assert isinstance(YDBChatAgentContextRepository(FakePool()), ChatAgentContextRepository)


async def test_save_upserts_the_row_of_the_chat():
    pool = FakePool()
    context = _recorded_context()

    saved = await YDBChatAgentContextRepository(pool).save(context)

    assert saved is context
    assert (
        "UPSERT INTO chat_agent_context (chat_id, created_at, last_date_field, last_project, "
        "last_tool, last_window_from, last_window_to, updated_at, user_id)"
    ) in pool.last_query
    assert "DECLARE $last_project AS Utf8?;" in pool.last_query
    assert pool.last_parameters["$last_project"].value == "Home renovation"
    assert pool.last_parameters["$chat_id"].value == str(context.chat_id)


async def test_save_writes_missing_helper_values_as_nulls():
    pool = FakePool()

    await YDBChatAgentContextRepository(pool).save(_context())

    for parameter in ("$last_tool", "$last_window_from", "$last_window_to", "$last_project"):
        assert pool.last_parameters[parameter].value is None


async def test_save_inside_a_transaction_runs_on_the_context_without_committing():
    pool, tx = FakePool(), FakeTx()

    await YDBChatAgentContextRepository(pool).save(_recorded_context(), tx=tx)

    assert pool.calls == []
    ((query, _parameters, commit_tx),) = tx.calls
    assert "UPSERT INTO chat_agent_context" in query
    assert commit_tx is False


async def test_find_by_chat_id_is_a_primary_key_read():
    context = _recorded_context()
    pool = FakePool(rows=[_row(context)])

    found = await YDBChatAgentContextRepository(pool).find_by_chat_id(context.chat_id)

    assert found == context
    query = one_line(pool.last_query)
    assert "FROM chat_agent_context WHERE chat_id = $chat_id;" in query
    assert "VIEW" not in query
    assert pool.last_parameters["$chat_id"].value == str(context.chat_id)


async def test_find_by_chat_id_returns_none_before_any_state_was_recorded():
    repository = YDBChatAgentContextRepository(FakePool())

    assert await repository.find_by_chat_id(ChatId.generate()) is None


class TestChatAgentContextMigration:
    def test_declares_the_table_nine_typed_columns_and_no_index(self):
        migration = load_agent_migration(MIGRATION_VERSION)
        artifacts = migration.get_artifacts()

        assert artifacts["tables"] == ["chat_agent_context"]
        assert artifacts.get("indexes", []) == []
        assert len(artifacts["columns"]) == 9
        assert declared_columns(migration, "chat_agent_context") == {
            "chat_id": "Utf8",
            "user_id": "Utf8",
            "last_tool": "Utf8?",
            "last_window_from": "Utf8?",
            "last_window_to": "Utf8?",
            "last_date_field": "Utf8?",
            "last_project": "Utf8?",
            "created_at": "Timestamp",
            "updated_at": "Timestamp",
        }

    def test_declared_columns_match_what_the_mapper_writes(self):
        declared = declared_columns(
            load_agent_migration(MIGRATION_VERSION), "chat_agent_context"
        )

        assert mapper_columns(ChatAgentContextMapper()) == declared

    async def test_creates_the_table_idempotently_keyed_by_chat(self):
        pool = FakePool()

        await load_agent_migration(MIGRATION_VERSION).up(pool)

        (statement,) = [one_line(query) for query, _parameters in pool.calls]
        assert statement.startswith("CREATE TABLE IF NOT EXISTS chat_agent_context (")
        assert "PRIMARY KEY (chat_id)" in statement
        assert "INDEX" not in statement
        assert "user_id Utf8 NOT NULL" in statement
        assert "last_project Utf8?," in statement
