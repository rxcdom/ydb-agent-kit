"""The agent module's repositories and migrations against a real YDB.

Covers what fakes cannot: the rendered YQL is accepted by the server, the named
indexes exist, ``Json`` and ``Utf8`` values survive the round trip, LIKE escaping
works, and a transaction really commits or rolls back.

The tables are the project's own, created by the migrations under test (they are
idempotent) and never dropped here. Every row written below belongs to a freshly
generated user or chat, and is removed when the test finishes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, List, Optional

import pytest
import ydb

from src.agent.adapters.persistence.ydb_agent_repository_manager import YDBAgentRepositoryManager
from src.agent.domain.entities.chat import Chat
from src.agent.domain.entities.chat_agent_context import ChatAgentContext
from src.agent.domain.entities.message import Message, MessageRole
from src.agent.domain.entities.user_memory import UserMemory
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.domain.value_objects.message_id import MessageId
from src.agent.domain.value_objects.message_status import MessageStatus
from src.shared.domain.value_objects.user_id import UserId
from src.shared.infrastructure.database.migration.discovery import discover_migration_files
from src.shared.infrastructure.database.migration.verification import verify_migration_artifacts
from src.shared.infrastructure.database.ydb.connection import YDBConnection

pytestmark = pytest.mark.integration

AGENT_MIGRATION_VERSIONS = ["20260901000100", "20260901000200", "20260901000300"]
BASE_TIME = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)

TRACE = {
    "model_name": "integration-test-model",
    "total_tokens": 57,
    "request_flow": [
        {
            "iteration": 1,
            "tokens": 40,
            "assistant_text": "",
            "function_calls": [
                {
                    "name": "query_tasks",
                    "call_id": "call-1",
                    "arguments": {"date_field": "due", "statuses": ["open"], "text": None},
                    "result": {
                        "status": "ok",
                        "data": {"total_count": 2, "note": "café — \"quoted\" 100%"},
                    },
                }
            ],
        },
        {"iteration": 2, "tokens": 17, "assistant_text": "Two tasks.", "function_calls": []},
    ],
}


@dataclass
class Workspace:
    """Freshly generated owners and chats, plus the clean-up of their rows."""

    manager: YDBAgentRepositoryManager
    pool: ydb.aio.QuerySessionPool
    user_ids: List[UserId] = field(default_factory=list)
    chat_ids: List[ChatId] = field(default_factory=list)

    def new_user(self) -> UserId:
        user_id = UserId.generate()
        self.user_ids.append(user_id)
        return user_id

    def new_chat(
        self, user_id: UserId, minutes: int = 0, title: Optional[str] = None
    ) -> Chat:
        chat = Chat(
            chat_id=ChatId.generate(),
            user_id=user_id,
            created_at=BASE_TIME + timedelta(minutes=minutes),
            updated_at=BASE_TIME + timedelta(minutes=minutes),
            title=title,
        )
        self.chat_ids.append(chat.chat_id)
        return chat

    async def clean_up(self) -> None:
        for chat_id in self.chat_ids:
            parameters = {"$chat_id": ydb.TypedValue(str(chat_id), ydb.PrimitiveType.Utf8)}
            for statement in (
                "DELETE FROM messages ON SELECT message_id FROM messages "
                "VIEW idx_messages_chat_created WHERE chat_id = $chat_id;",
                "DELETE FROM chat_agent_context WHERE chat_id = $chat_id;",
                "DELETE FROM chats WHERE chat_id = $chat_id;",
            ):
                await self.pool.execute_with_retries(
                    f"DECLARE $chat_id AS Utf8; {statement}", parameters=parameters
                )
        for user_id in self.user_ids:
            await self.manager.user_memory.delete_all_by_user_id(user_id)


@pytest.fixture
async def workspace(ydb_connection: YDBConnection) -> AsyncIterator[Workspace]:
    for migration_file in _agent_migration_files():
        await migration_file.import_class()().up(ydb_connection.pool)

    space = Workspace(
        manager=YDBAgentRepositoryManager(ydb_connection.pool), pool=ydb_connection.pool
    )
    try:
        yield space
    finally:
        await space.clean_up()


def _agent_migration_files():
    files = [f for f in discover_migration_files() if f.module_name == "agent"]
    assert [f.version for f in files] == AGENT_MIGRATION_VERSIONS
    return files


def _message(
    chat: Chat,
    minutes: int,
    role: MessageRole = MessageRole.USER,
    content: str = "What is overdue?",
    status: MessageStatus = MessageStatus.SENT,
    tokens: int = 0,
    trace: Optional[dict] = None,
) -> Message:
    return Message(
        message_id=MessageId.generate(),
        chat_id=chat.chat_id,
        user_id=chat.user_id,
        role=role,
        content=content,
        tokens=tokens,
        status=status,
        created_at=chat.created_at + timedelta(minutes=minutes),
        trace=trace,
    )


def _memory(
    user_id: UserId, content: str, topic: Optional[str], minutes: int, updated_minutes: int
) -> UserMemory:
    return UserMemory(
        memory_id=UserMemory.generate_id(),
        user_id=user_id,
        content=content,
        topic=topic,
        created_at=BASE_TIME + timedelta(minutes=minutes),
        updated_at=BASE_TIME + timedelta(minutes=updated_minutes),
    )


async def test_migrations_are_idempotent_and_their_declarations_match_the_live_schema(
    ydb_connection: YDBConnection, workspace: Workspace
) -> None:
    for migration_file in _agent_migration_files():
        migration = migration_file.import_class()()
        # The fixture applied it once already; a second run has to be a no-op.
        await migration.up(ydb_connection.pool)
        await verify_migration_artifacts(
            ydb_connection.driver, ydb_connection.database, migration, migration_file
        )


async def test_chats_are_saved_found_and_listed_per_owner(workspace: Workspace) -> None:
    chats = workspace.manager.chats
    owner, other_owner = workspace.new_user(), workspace.new_user()
    older = workspace.new_chat(owner, minutes=0, title="Weekly planning")
    newer = workspace.new_chat(owner, minutes=5)
    foreign = workspace.new_chat(other_owner, minutes=3, title="Somebody else")
    for chat in (older, newer, foreign):
        await chats.save(chat)

    assert await chats.find_by_id(older.chat_id) == older
    assert (await chats.find_by_id(newer.chat_id)).title is None
    assert await chats.find_by_id(ChatId.generate()) is None
    assert await chats.find_by_user_id(owner) == [newer, older]
    assert await chats.find_by_user_id(other_owner) == [foreign]
    assert await chats.find_by_user_id(UserId.generate()) == []

    older.title = "Renamed"
    older.updated_at = older.updated_at + timedelta(hours=1)
    await chats.save(older)

    assert await chats.find_by_id(older.chat_id) == older
    assert len(await chats.find_by_user_id(owner)) == 2


async def test_messages_keep_their_order_pages_and_tail(workspace: Workspace) -> None:
    messages = workspace.manager.messages
    chat = workspace.new_chat(workspace.new_user())
    other_chat = workspace.new_chat(chat.user_id, minutes=1)
    stored = [_message(chat, minutes=index, content=f"Message {index}") for index in range(5)]
    # Saved out of order: the order has to come from created_at, not from insertion.
    for message in (stored[3], stored[0], stored[4], stored[1], stored[2]):
        await messages.save(message)
    await messages.save(_message(other_chat, minutes=0, content="Another chat"))

    assert await messages.find_by_chat_id(chat.chat_id) == stored
    assert await messages.find_by_chat_id(chat.chat_id, limit=2) == stored[:2]
    assert await messages.find_by_chat_id(chat.chat_id, limit=2, offset=2) == stored[2:4]
    assert await messages.find_by_chat_id(chat.chat_id, limit=10, offset=4) == stored[4:]
    assert await messages.find_by_chat_id(chat.chat_id, limit=10, offset=5) == []

    assert await messages.find_recent_by_chat_id(chat.chat_id, 3) == stored[2:]
    assert await messages.find_recent_by_chat_id(chat.chat_id, 50) == stored

    assert await messages.count_by_chat_id(chat.chat_id) == 5
    assert await messages.count_by_chat_id(other_chat.chat_id) == 1
    assert await messages.count_by_chat_id(ChatId.generate()) == 0
    assert await messages.find_by_chat_id(ChatId.generate()) == []


async def test_messages_with_the_same_timestamp_page_without_gaps_or_repeats(
    workspace: Workspace,
) -> None:
    messages = workspace.manager.messages
    chat = workspace.new_chat(workspace.new_user())
    simultaneous = [_message(chat, minutes=0, content=f"Same instant {index}") for index in range(4)]
    for message in simultaneous:
        await messages.save(message)

    pages = [
        await messages.find_by_chat_id(chat.chat_id, limit=2, offset=offset) for offset in (0, 2)
    ]
    paged = [message for page in pages for message in page]

    assert sorted(str(m.message_id) for m in paged) == sorted(
        str(m.message_id) for m in simultaneous
    )
    assert paged == await messages.find_by_chat_id(chat.chat_id)
    assert await messages.find_recent_by_chat_id(chat.chat_id, 2) == paged[2:]


async def test_trace_survives_the_json_round_trip(workspace: Workspace) -> None:
    messages = workspace.manager.messages
    chat = workspace.new_chat(workspace.new_user())
    question = _message(chat, minutes=0)
    reply = _message(
        chat,
        minutes=1,
        role=MessageRole.ASSISTANT,
        content="Two tasks are overdue.",
        tokens=57,
        trace=TRACE,
    )
    await messages.save(question)
    await messages.save(reply)

    restored_question, restored_reply = await messages.find_by_chat_id(chat.chat_id)

    assert restored_question.trace is None
    assert restored_reply == reply
    assert restored_reply.trace == TRACE
    assert restored_reply.role is MessageRole.ASSISTANT
    assert restored_reply.tokens == 57


async def test_transaction_commits_the_status_update_together_with_the_reply(
    workspace: Workspace,
) -> None:
    manager = workspace.manager
    chat = workspace.new_chat(workspace.new_user())
    question = _message(chat, minutes=0, status=MessageStatus.PROCESSING)
    await manager.messages.save(question)
    reply = _message(
        chat,
        minutes=1,
        role=MessageRole.ASSISTANT,
        content="Two tasks are overdue.",
        tokens=57,
        trace=TRACE,
    )

    results = await manager.execute_in_transaction(
        [
            lambda tx: manager.messages.update_status(
                question.message_id, MessageStatus.SENT, tx=tx
            ),
            lambda tx: manager.messages.save(reply, tx=tx),
        ]
    )

    assert results == [None, reply]
    stored_question, stored_reply = await manager.messages.find_by_chat_id(chat.chat_id)
    assert stored_question.message_id == question.message_id
    assert stored_question.status is MessageStatus.SENT
    # The status update leaves every other field of the row as it was.
    assert stored_question.content == question.content
    assert stored_question.created_at == question.created_at
    assert stored_reply == reply


async def test_failed_transaction_leaves_no_trace_of_its_writes(workspace: Workspace) -> None:
    manager = workspace.manager
    chat = workspace.new_chat(workspace.new_user())
    question = _message(chat, minutes=0, status=MessageStatus.PROCESSING)
    await manager.messages.save(question)
    reply = _message(chat, minutes=1, role=MessageRole.ASSISTANT, content="Never stored.")
    context = ChatAgentContext.empty_for(chat.chat_id, chat.user_id)

    async def failing(_tx):
        raise RuntimeError("the turn failed after its writes")

    with pytest.raises(RuntimeError, match="the turn failed"):
        await manager.execute_in_transaction(
            [
                lambda tx: manager.messages.update_status(
                    question.message_id, MessageStatus.SENT, tx=tx
                ),
                lambda tx: manager.messages.save(reply, tx=tx),
                lambda tx: manager.chat_agent_context.save(context, tx=tx),
                failing,
            ]
        )

    (only_message,) = await manager.messages.find_by_chat_id(chat.chat_id)
    assert only_message == question
    assert only_message.status is MessageStatus.PROCESSING
    assert await manager.messages.count_by_chat_id(chat.chat_id) == 1
    assert await manager.chat_agent_context.find_by_chat_id(chat.chat_id) is None


async def test_status_update_outside_a_transaction(workspace: Workspace) -> None:
    messages = workspace.manager.messages
    chat = workspace.new_chat(workspace.new_user())
    question = _message(chat, minutes=0, status=MessageStatus.PROCESSING)
    await messages.save(question)

    await messages.update_status(question.message_id, MessageStatus.FAILED)
    await messages.update_status(MessageId.generate(), MessageStatus.FAILED)

    (stored,) = await messages.find_by_chat_id(chat.chat_id)
    assert stored.status is MessageStatus.FAILED
    assert await messages.count_by_chat_id(chat.chat_id) == 1


async def test_memory_search_is_case_insensitive_literal_and_owner_scoped(
    workspace: Workspace,
) -> None:
    memory = workspace.manager.user_memory
    owner, other_owner = workspace.new_user(), workspace.new_user()
    cotton = _memory(owner, "Prefers 100% cotton_shirts", "Clothes", 0, updated_minutes=10)
    steps = _memory(owner, "Walks 1000 steps before breakfast", None, 1, updated_minutes=20)
    hyphen = _memory(owner, "Owns two cotton-shirts in blue", "wardrobe", 2, updated_minutes=30)
    birthday = _memory(owner, "Throws a party every spring", "Birthday", 3, updated_minutes=40)
    foreign = _memory(other_owner, "Also likes COTTON shirts", "clothes", 4, updated_minutes=50)
    for entry in (cotton, steps, hyphen, birthday, foreign):
        await memory.save(entry)

    async def search(query: str, limit: int = 10) -> List[UserMemory]:
        return await memory.search_by_user_id(owner, query=query, limit=limit)

    # Case-insensitive on content, freshest first, never another owner's rows.
    assert await search("COTTON") == [hyphen, cotton]
    assert await search("COTTON", limit=1) == [hyphen]
    # Case-insensitive on the topic; a row without a topic is matched on content alone.
    assert await search("birthday") == [birthday]
    assert await search("BREAKFAST") == [steps]
    # "%" and "_" are literals: "1000 steps" holds no "100%", "cotton-shirts" no "cotton_shirts".
    assert await search("100%") == [cotton]
    assert await search("cotton_shirts") == [cotton]
    assert await search("no such fact") == []
    assert await memory.search_by_user_id(other_owner, query="cotton", limit=10) == [foreign]


async def test_memory_listing_count_oldest_and_upsert(workspace: Workspace) -> None:
    memory = workspace.manager.user_memory
    owner = workspace.new_user()
    first = _memory(owner, "Works from home on Fridays", "schedule", 0, updated_minutes=30)
    second = _memory(owner, "Allergic to peanuts", "health", 1, updated_minutes=10)
    third = _memory(owner, "Learns Spanish", None, 2, updated_minutes=20)

    assert await memory.count_by_user_id(owner) == 0
    assert await memory.find_oldest_by_user_id(owner) is None
    assert await memory.find_by_user_id(owner, limit=10) == []

    for entry in (second, third, first):
        await memory.save(entry)

    assert await memory.count_by_user_id(owner) == 3
    assert await memory.find_oldest_by_user_id(owner) == first
    assert await memory.find_by_user_id(owner, limit=10) == [first, third, second]
    assert await memory.find_by_user_id(owner, limit=2) == [first, third]

    second.content = "Allergic to peanuts and shellfish"
    second.updated_at = BASE_TIME + timedelta(minutes=60)
    await memory.save(second)

    assert await memory.count_by_user_id(owner) == 3
    assert await memory.find_by_user_id(owner, limit=1) == [second]


async def test_memory_delete_is_guarded_by_the_owner(workspace: Workspace) -> None:
    memory = workspace.manager.user_memory
    owner, other_owner = workspace.new_user(), workspace.new_user()
    kept = _memory(owner, "Keeps a standing desk", "office", 0, updated_minutes=0)
    removed = _memory(owner, "Plans to sell the old bike", "bike", 1, updated_minutes=1)
    foreign = _memory(other_owner, "Rides to work", "bike", 2, updated_minutes=2)
    for entry in (kept, removed, foreign):
        await memory.save(entry)

    # Another owner cannot delete the row, even with the right memory id.
    await memory.delete(removed.memory_id, other_owner)
    assert await memory.count_by_user_id(owner) == 2

    await memory.delete(removed.memory_id, owner)
    assert await memory.find_by_user_id(owner, limit=10) == [kept]

    await memory.delete_all_by_user_id(owner)
    assert await memory.count_by_user_id(owner) == 0
    assert await memory.find_by_user_id(other_owner, limit=10) == [foreign]


async def test_conversation_state_is_upserted_per_chat(workspace: Workspace) -> None:
    contexts = workspace.manager.chat_agent_context
    chat = workspace.new_chat(workspace.new_user())

    assert await contexts.find_by_chat_id(chat.chat_id) is None

    context = ChatAgentContext(
        chat_id=chat.chat_id,
        user_id=chat.user_id,
        created_at=BASE_TIME,
        updated_at=BASE_TIME,
    )
    await contexts.save(context)
    stored = await contexts.find_by_chat_id(chat.chat_id)
    assert stored == context
    assert stored.has_context() is False

    context.record_windowed_tool(
        tool="query_tasks",
        window_from="2026-09-01",
        window_to="2026-09-17",
        date_field="due",
        project="Home renovation",
    )
    await contexts.save(context)

    # The entity stamped a new update time with microseconds; they survive the round trip.
    stored = await contexts.find_by_chat_id(chat.chat_id)
    assert stored == context
    assert stored.updated_at > BASE_TIME
    assert stored.last_project == "Home renovation"

    context.record_windowed_tool(
        tool="query_tasks",
        window_from="2026-08-01",
        window_to="2026-08-31",
        date_field="created",
        project=None,
    )
    await contexts.save(context)

    stored = await contexts.find_by_chat_id(chat.chat_id)
    assert stored == context
    assert stored.last_project is None
    assert stored.last_window_from == "2026-08-01"
