"""Creates the ``user_memory`` table: the agent's long-term memory vault.

One row is one durable fact a user asked the agent to keep. Rows are read on
demand by the ``recall`` tool; the system prompt only ever carries a pointer
(the row count and the topics). Every read is scoped by the owner, so the owner
id carries ``idx_user_memory_user_id``. ``topic`` is an optional short tag.
"""
from __future__ import annotations

import ydb

from src.shared.infrastructure.database.migration.base import Migration
from src.shared.infrastructure.database.migration.logging import log_error, log_table_creation


class CreateUserMemoryTableMigration(Migration):
    version = "20260901000200"
    description = "Create user_memory table"

    def get_artifacts(self):
        return {
            "tables": ["user_memory"],
            "indexes": [("user_memory", "idx_user_memory_user_id")],
            "columns": [
                ("user_memory", "memory_id", "Utf8"),
                ("user_memory", "user_id", "Utf8"),
                ("user_memory", "content", "Utf8"),
                ("user_memory", "topic", "Utf8?"),
                ("user_memory", "created_at", "Timestamp"),
                ("user_memory", "updated_at", "Timestamp"),
            ],
        }

    async def up(self, pool: ydb.aio.QuerySessionPool) -> None:
        log_table_creation("user_memory", "creating")
        try:
            await pool.execute_with_retries(
                """
                CREATE TABLE IF NOT EXISTS user_memory (
                    memory_id  Utf8 NOT NULL,
                    user_id    Utf8 NOT NULL,
                    content    Utf8 NOT NULL,
                    topic      Utf8?,
                    created_at Timestamp NOT NULL,
                    updated_at Timestamp NOT NULL,
                    PRIMARY KEY (memory_id),
                    INDEX idx_user_memory_user_id GLOBAL ON (user_id)
                );
                """
            )
        except ydb.Error as error:
            log_table_creation("user_memory", "error")
            log_error(f"Error: {error}", indent=3)
            raise
        log_table_creation("user_memory", "success")
