import json
from datetime import datetime, timedelta, timezone

import pytest
import ydb

from src.agent.adapters.persistence.ydb_message_repository import (
    MessageMapper,
    YDBMessageRepository,
)
from src.agent.domain.entities.message import Message, MessageRole
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.domain.value_objects.message_id import MessageId
from src.agent.domain.value_objects.message_status import MessageStatus
from src.agent.ports.message_repository import MessageRepository
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
BASE_TIME = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)

TRACE = {
    "model_name": "unit-test-model",
    "total_tokens": 42,
    "request_flow": [
        {
            "iteration": 1,
            "tokens": 30,
            "assistant_text": "",
            "function_calls": [
                {
                    "name": "query_tasks",
                    "call_id": "call-1",
                    "arguments": {"date_field": "due", "statuses": ["open"]},
                    "result": {"status": "ok", "data": {"total_count": 2}},
                }
            ],
        }
    ],
}


def _message(role=MessageRole.USER, trace=None, minutes=0, content="What is overdue?") -> Message:
    return Message(
        message_id=MessageId.generate(),
        chat_id=ChatId.generate(),
        user_id=UserId.generate(),
        role=role,
        content=content,
        tokens=42 if role is MessageRole.ASSISTANT else 0,
        status=MessageStatus.SENT if role is MessageRole.ASSISTANT else MessageStatus.PROCESSING,
        created_at=BASE_TIME + timedelta(minutes=minutes),
        trace=trace,
    )


def _row(message: Message, **overrides) -> dict:
    params = MessageMapper().to_ydb_params(message)
    row = {name.removeprefix("$"): value for name, value in params.items()}
    row["created_at"] = row["created_at"].replace(tzinfo=None)
    row.update(overrides)
    return row


class TestMapper:
    def test_round_trips_a_user_message_without_a_trace(self):
        message = _message()

        assert MessageMapper().to_domain(_row(message)) == message

    def test_round_trips_an_assistant_message_with_its_trace(self):
        message = _message(MessageRole.ASSISTANT, trace=TRACE, content="Two tasks are overdue.")

        assert MessageMapper().to_domain(_row(message)) == message

    def test_trace_is_written_as_json_text_and_read_back_from_every_wire_form(self):
        message = _message(MessageRole.ASSISTANT, trace=TRACE, content="Two tasks are overdue.")
        repository = YDBMessageRepository(FakePool())

        typed = repository._build_ydb_params(MessageMapper().to_ydb_params(message))

        written = typed["$trace"].value
        assert isinstance(written, str)
        assert json.loads(written) == TRACE
        assert isinstance(typed["$trace"].value_type, ydb.OptionalType)
        for wire_value in (written, written.encode("utf-8"), TRACE):
            assert MessageMapper().to_domain(_row(message, trace=wire_value)).trace == TRACE

    def test_trace_that_is_not_an_object_is_rejected(self):
        with pytest.raises(ValueError, match="JSON object"):
            MessageMapper().to_domain(_row(_message(), trace="[1, 2]"))

    @pytest.mark.parametrize("column, value", [("role", "system"), ("status", "archived")])
    def test_unknown_role_or_status_is_rejected(self, column, value):
        with pytest.raises(ValueError, match=value):
            MessageMapper().to_domain(_row(_message(), **{column: value}))

    def test_column_parameters_exclude_the_query_only_ones(self):
        mapper = MessageMapper()

        assert set(mapper.get_column_params()) == set(mapper.to_ydb_params(_message()))
        assert {"$limit", "$offset"} <= set(mapper.get_ydb_type_map())
        assert not {"$limit", "$offset"} & set(mapper.get_column_params())


def test_repository_implements_the_port():
    assert isinstance(YDBMessageRepository(FakePool()), MessageRepository)


class TestSave:
    async def test_upserts_every_column(self):
        pool = FakePool()
        message = _message(MessageRole.ASSISTANT, trace=TRACE, content="Two tasks are overdue.")

        saved = await YDBMessageRepository(pool).save(message)

        assert saved is message
        assert (
            "UPSERT INTO messages (chat_id, content, created_at, message_id, role, status, "
            "tokens, trace, user_id)"
        ) in pool.last_query
        assert "DECLARE $trace AS Json?;" in pool.last_query
        assert "DECLARE $tokens AS Int32;" in pool.last_query
        assert "$limit" not in pool.last_query
        assert pool.last_parameters["$role"].value == "assistant"
        assert pool.last_parameters["$status"].value == "sent"
        assert pool.last_parameters["$tokens"].value == 42

    async def test_inside_a_transaction_runs_on_the_context_without_committing(self):
        pool, tx = FakePool(), FakeTx()

        await YDBMessageRepository(pool).save(_message(), tx=tx)

        assert pool.calls == []
        ((query, parameters, commit_tx),) = tx.calls
        assert "UPSERT INTO messages" in query
        assert parameters["$trace"].value is None
        assert commit_tx is False


class TestFindByChatId:
    async def test_names_the_chat_index_and_orders_oldest_first(self):
        first, second = _message(minutes=0), _message(minutes=1)
        pool = FakePool(rows=[_row(first), _row(second)])
        chat_id = ChatId.generate()

        messages = await YDBMessageRepository(pool).find_by_chat_id(chat_id)

        assert messages == [first, second]
        query = one_line(pool.last_query)
        assert "FROM messages VIEW idx_messages_chat_created WHERE chat_id = $chat_id" in query
        assert "ORDER BY created_at ASC, message_id ASC" in query
        assert pool.last_parameters["$chat_id"].value == str(chat_id)

    async def test_without_a_limit_no_limit_clause_is_rendered(self):
        pool = FakePool()

        await YDBMessageRepository(pool).find_by_chat_id(ChatId.generate())

        assert "LIMIT" not in pool.last_query
        assert "OFFSET" not in pool.last_query
        assert set(pool.last_parameters) == {"$chat_id"}
        assert one_line(pool.last_query).endswith("ORDER BY created_at ASC, message_id ASC;")

    async def test_limit_and_offset_are_typed_parameters_not_inlined_values(self):
        pool = FakePool()

        await YDBMessageRepository(pool).find_by_chat_id(ChatId.generate(), limit=50, offset=100)

        query = one_line(pool.last_query)
        assert query.endswith("LIMIT $limit OFFSET $offset;")
        assert "DECLARE $limit AS Uint64;" in query
        assert "DECLARE $offset AS Uint64;" in query
        assert "50" not in query and "100" not in query
        assert pool.last_parameters["$limit"].value == 50
        assert pool.last_parameters["$offset"].value == 100

    async def test_limit_alone_pages_from_the_first_message(self):
        pool = FakePool()

        await YDBMessageRepository(pool).find_by_chat_id(ChatId.generate(), limit=10)

        assert one_line(pool.last_query).endswith("LIMIT $limit;")
        assert "$offset" not in pool.last_parameters

    async def test_offset_without_a_limit_is_rejected_before_any_query(self):
        pool = FakePool()

        with pytest.raises(ValueError, match="offset requires a limit"):
            await YDBMessageRepository(pool).find_by_chat_id(ChatId.generate(), offset=5)

        assert pool.calls == []

    async def test_inside_a_transaction_reads_through_the_context(self):
        message = _message()
        pool, tx = FakePool(), FakeTx(rows=[_row(message)])

        messages = await YDBMessageRepository(pool).find_by_chat_id(message.chat_id, tx=tx)

        assert messages == [message]
        assert pool.calls == []
        assert tx.calls[0][2] is False


class TestFindRecentByChatId:
    async def test_selects_the_newest_rows_and_returns_them_oldest_first(self):
        oldest, middle, newest = (_message(minutes=index) for index in range(3))
        # The database answers newest first.
        pool = FakePool(rows=[_row(newest), _row(middle), _row(oldest)])

        recent = await YDBMessageRepository(pool).find_recent_by_chat_id(ChatId.generate(), 3)

        assert recent == [oldest, middle, newest]
        query = one_line(pool.last_query)
        assert "FROM messages VIEW idx_messages_chat_created WHERE chat_id = $chat_id" in query
        assert query.endswith("ORDER BY created_at DESC, message_id DESC LIMIT $limit;")
        assert pool.last_parameters["$limit"].value == 3


class TestCountByChatId:
    async def test_counts_through_the_chat_index(self):
        pool = FakePool(rows=[{"total": 7}])

        total = await YDBMessageRepository(pool).count_by_chat_id(ChatId.generate())

        assert total == 7
        query = one_line(pool.last_query)
        assert "SELECT COUNT(*) AS total FROM messages VIEW idx_messages_chat_created" in query
        assert "WHERE chat_id = $chat_id" in query

    async def test_no_result_row_counts_as_zero(self):
        assert await YDBMessageRepository(FakePool()).count_by_chat_id(ChatId.generate()) == 0


class TestUpdateStatus:
    async def test_updates_one_row_by_primary_key(self):
        pool = FakePool()
        message_id = MessageId.generate()

        await YDBMessageRepository(pool).update_status(message_id, MessageStatus.FAILED)

        query = one_line(pool.last_query)
        assert "UPDATE messages SET status = $status WHERE message_id = $message_id;" in query
        assert "VIEW" not in query
        assert pool.last_parameters["$message_id"].value == str(message_id)
        assert pool.last_parameters["$status"].value == "failed"

    async def test_inside_a_transaction_runs_on_the_context_without_committing(self):
        pool, tx = FakePool(), FakeTx()

        await YDBMessageRepository(pool).update_status(
            MessageId.generate(), MessageStatus.SENT, tx=tx
        )

        assert pool.calls == []
        ((query, parameters, commit_tx),) = tx.calls
        assert "UPDATE messages" in query
        assert parameters["$status"].value == "sent"
        assert commit_tx is False


class TestMessagesMigration:
    def test_messages_columns_match_the_planned_schema_and_the_mapper(self):
        declared = declared_columns(load_agent_migration(MIGRATION_VERSION), "messages")

        assert declared == {
            "message_id": "Utf8",
            "chat_id": "Utf8",
            "user_id": "Utf8",
            "role": "Utf8",
            "content": "Utf8",
            "tokens": "Int32",
            "status": "Utf8",
            "trace": "Json?",
            "created_at": "Timestamp",
        }
        assert mapper_columns(MessageMapper()) == declared

    async def test_creates_the_messages_table_idempotently_with_its_inline_index(self):
        pool = FakePool()

        await load_agent_migration(MIGRATION_VERSION).up(pool)

        assert len(pool.calls) == 2
        statement = one_line(pool.calls[1][0])
        assert statement.startswith("CREATE TABLE IF NOT EXISTS messages (")
        assert "PRIMARY KEY (message_id)" in statement
        assert "INDEX idx_messages_chat_created GLOBAL ON (chat_id, created_at)" in statement
        assert "tokens Int32 NOT NULL" in statement
        assert "trace Json?," in statement
