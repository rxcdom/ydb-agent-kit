"""Creates the ``chats`` and ``messages`` tables.

A chat belongs to one user and is listed through ``idx_chats_user_id``. A message
repeats the owner id of its chat, so ownership can be checked on the row itself.
Messages are read per chat in time order, which is the key of
``idx_messages_chat_created``. ``trace`` holds the debug trace of the agent loop
for an assistant message and is empty for a user message.

Required columns are ``NOT NULL``: the schema inspection reports every other
column as optional, and the declared artifact types have to match it.
"""
from __future__ import annotations

import ydb

from src.shared.infrastructure.database.migration.base import Migration
from src.shared.infrastructure.database.migration.logging import log_error, log_table_creation

_CREATE_CHATS = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id    Utf8 NOT NULL,
    user_id    Utf8 NOT NULL,
    title      Utf8?,
    created_at Timestamp NOT NULL,
    updated_at Timestamp NOT NULL,
    PRIMARY KEY (chat_id),
    INDEX idx_chats_user_id GLOBAL ON (user_id)
);
"""

_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    message_id Utf8 NOT NULL,
    chat_id    Utf8 NOT NULL,
    user_id    Utf8 NOT NULL,
    role       Utf8 NOT NULL,
    content    Utf8 NOT NULL,
    tokens     Int32 NOT NULL,
    status     Utf8 NOT NULL,
    trace      Json?,
    created_at Timestamp NOT NULL,
    PRIMARY KEY (message_id),
    INDEX idx_messages_chat_created GLOBAL ON (chat_id, created_at)
);
"""


async def _create_table(pool: ydb.aio.QuerySessionPool, table_name: str, statement: str) -> None:
    log_table_creation(table_name, "creating")
    try:
        await pool.execute_with_retries(statement)
    except ydb.Error as error:
        log_table_creation(table_name, "error")
        log_error(f"Error: {error}", indent=3)
        raise
    log_table_creation(table_name, "success")


class CreateChatsAndMessagesMigration(Migration):
    version = "20260901000100"
    description = "Create chats and messages tables"

    def get_artifacts(self):
        return {
            "tables": ["chats", "messages"],
            "indexes": [
                ("chats", "idx_chats_user_id"),
                ("messages", "idx_messages_chat_created"),
            ],
            "columns": [
                ("chats", "chat_id", "Utf8"),
                ("chats", "user_id", "Utf8"),
                ("chats", "title", "Utf8?"),
                ("chats", "created_at", "Timestamp"),
                ("chats", "updated_at", "Timestamp"),
                ("messages", "message_id", "Utf8"),
                ("messages", "chat_id", "Utf8"),
                ("messages", "user_id", "Utf8"),
                ("messages", "role", "Utf8"),
                ("messages", "content", "Utf8"),
                ("messages", "tokens", "Int32"),
                ("messages", "status", "Utf8"),
                ("messages", "trace", "Json?"),
                ("messages", "created_at", "Timestamp"),
            ],
        }

    async def up(self, pool: ydb.aio.QuerySessionPool) -> None:
        await _create_table(pool, "chats", _CREATE_CHATS)
        await _create_table(pool, "messages", _CREATE_MESSAGES)
