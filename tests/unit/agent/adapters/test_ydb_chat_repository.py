from datetime import datetime, timezone

import pytest
import ydb

from src.agent.adapters.persistence.ydb_chat_repository import ChatMapper, YDBChatRepository
from src.agent.domain.entities.chat import Chat
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.ports.chat_repository import ChatRepository
from src.shared.domain.exceptions import PersistenceError
from src.shared.domain.value_objects.user_id import UserId
from tests.unit.agent.adapters.ydb_fakes import (
    FakePool,
    FakeTx,
    declared_columns,
    load_agent_migration,
    mapper_columns,
    one_line,
)

MIGRATION_VERSION = "20260901000100"


def _chat(title="Weekly planning") -> Chat:
    return Chat(
        chat_id=ChatId.generate(),
        user_id=UserId.generate(),
        created_at=datetime(2026, 9, 1, 8, 30, tzinfo=timezone.utc),
        updated_at=datetime(2026, 9, 2, 9, 45, tzinfo=timezone.utc),
        title=title,
    )


def _row(chat: Chat) -> dict:
    params = ChatMapper().to_ydb_params(chat)
    row = {name.removeprefix("$"): value for name, value in params.items()}
    # The SDK returns naive UTC datetimes for Timestamp columns.
    row["created_at"] = row["created_at"].replace(tzinfo=None)
    row["updated_at"] = row["updated_at"].replace(tzinfo=None)
    return row


@pytest.mark.parametrize("title", ["Weekly planning", None])
def test_mapper_round_trips_a_chat(title):
    chat = _chat(title)

    assert ChatMapper().to_domain(_row(chat)) == chat


def test_mapper_reports_a_missing_column():
    row = _row(_chat())
    del row["title"]

    with pytest.raises(ValueError, match="title"):
        ChatMapper().to_domain(row)


def test_repository_implements_the_port():
    assert isinstance(YDBChatRepository(FakePool()), ChatRepository)


async def test_save_upserts_every_column_with_typed_parameters():
    pool = FakePool()
    chat = _chat(title=None)

    saved = await YDBChatRepository(pool).save(chat)

    assert saved is chat
    assert "UPSERT INTO chats (chat_id, created_at, title, updated_at, user_id)" in pool.last_query
    parameters = pool.last_parameters
    assert parameters["$chat_id"].value == str(chat.chat_id)
    assert parameters["$chat_id"].value_type == ydb.PrimitiveType.Utf8
    assert parameters["$title"].value is None
    assert parameters["$created_at"].value == chat.created_at


async def test_save_inside_a_transaction_runs_on_the_context_without_committing():
    pool, tx = FakePool(), FakeTx()

    await YDBChatRepository(pool).save(_chat(), tx=tx)

    assert pool.calls == []
    ((query, _parameters, commit_tx),) = tx.calls
    assert "UPSERT INTO chats" in query
    assert commit_tx is False


async def test_find_by_id_is_a_primary_key_read():
    chat = _chat()
    pool = FakePool(rows=[_row(chat)])

    found = await YDBChatRepository(pool).find_by_id(chat.chat_id)

    assert found == chat
    query = one_line(pool.last_query)
    assert "FROM chats WHERE chat_id = $chat_id" in query
    assert "VIEW" not in query
    assert pool.last_parameters["$chat_id"].value == str(chat.chat_id)


async def test_find_by_id_returns_none_for_an_unknown_chat():
    assert await YDBChatRepository(FakePool()).find_by_id(ChatId.generate()) is None


async def test_find_by_user_id_names_the_owner_index_and_lists_newest_first():
    first, second = _chat("First"), _chat("Second")
    pool = FakePool(rows=[_row(second), _row(first)])
    user_id = UserId.generate()

    chats = await YDBChatRepository(pool).find_by_user_id(user_id)

    assert chats == [second, first]
    query = one_line(pool.last_query)
    assert "FROM chats VIEW idx_chats_user_id WHERE user_id = $user_id" in query
    assert "ORDER BY created_at DESC" in query
    assert "DECLARE $user_id AS Utf8;" in query
    assert pool.last_parameters["$user_id"].value == str(user_id)


async def test_sdk_failure_surfaces_as_persistence_error():
    pool = FakePool(error=ydb.issues.Unavailable("node is down"))

    with pytest.raises(PersistenceError, match="chats"):
        await YDBChatRepository(pool).find_by_user_id(UserId.generate())


class TestChatsMigration:
    def test_declares_both_tables_both_indexes_and_fourteen_typed_columns(self):
        artifacts = load_agent_migration(MIGRATION_VERSION).get_artifacts()

        assert artifacts["tables"] == ["chats", "messages"]
        assert artifacts["indexes"] == [
            ("chats", "idx_chats_user_id"),
            ("messages", "idx_messages_chat_created"),
        ]
        assert len(artifacts["columns"]) == 14
        assert all(len(column) == 3 for column in artifacts["columns"])

    def test_chats_columns_match_the_planned_schema_and_the_mapper(self):
        declared = declared_columns(load_agent_migration(MIGRATION_VERSION), "chats")

        assert declared == {
            "chat_id": "Utf8",
            "user_id": "Utf8",
            "title": "Utf8?",
            "created_at": "Timestamp",
            "updated_at": "Timestamp",
        }
        assert mapper_columns(ChatMapper()) == declared

    async def test_creates_the_chats_table_idempotently_with_its_inline_index(self):
        pool = FakePool()

        await load_agent_migration(MIGRATION_VERSION).up(pool)

        chats_statement = one_line(pool.calls[0][0])
        assert chats_statement.startswith("CREATE TABLE IF NOT EXISTS chats (")
        assert "PRIMARY KEY (chat_id)" in chats_statement
        assert "INDEX idx_chats_user_id GLOBAL ON (user_id)" in chats_statement
        # Required columns are NOT NULL, so the live schema reports them as declared.
        for column in ("chat_id Utf8", "user_id Utf8", "created_at Timestamp"):
            assert f"{column} NOT NULL" in chats_statement
        assert "title Utf8?," in chats_statement

    async def test_failed_statement_is_reported_and_re_raised(self, capsys):
        pool = FakePool(error=ydb.issues.SchemeError("access denied"))

        with pytest.raises(ydb.issues.SchemeError):
            await load_agent_migration(MIGRATION_VERSION).up(pool)

        assert len(pool.calls) == 1
        assert "access denied" in capsys.readouterr().err
