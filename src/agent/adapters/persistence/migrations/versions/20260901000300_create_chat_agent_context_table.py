"""Creates the ``chat_agent_context`` table: the per-chat conversation state.

One row per chat keeps the window, the date axis, the project and the tool of the
last windowed query, so a follow-up question can reuse them. The row is derived
state and is always read by its primary key, so the table has no secondary index.
Every helper value is optional: a chat may have no state yet.
"""
from __future__ import annotations

import ydb

from src.shared.infrastructure.database.migration.base import Migration
from src.shared.infrastructure.database.migration.logging import log_error, log_table_creation


class CreateChatAgentContextTableMigration(Migration):
    version = "20260901000300"
    description = "Create chat_agent_context table"

    def get_artifacts(self):
        return {
            "tables": ["chat_agent_context"],
            "columns": [
                ("chat_agent_context", "chat_id", "Utf8"),
                ("chat_agent_context", "user_id", "Utf8"),
                ("chat_agent_context", "last_tool", "Utf8?"),
                ("chat_agent_context", "last_window_from", "Utf8?"),
                ("chat_agent_context", "last_window_to", "Utf8?"),
                ("chat_agent_context", "last_date_field", "Utf8?"),
                ("chat_agent_context", "last_project", "Utf8?"),
                ("chat_agent_context", "created_at", "Timestamp"),
                ("chat_agent_context", "updated_at", "Timestamp"),
            ],
        }

    async def up(self, pool: ydb.aio.QuerySessionPool) -> None:
        log_table_creation("chat_agent_context", "creating")
        try:
            await pool.execute_with_retries(
                """
                CREATE TABLE IF NOT EXISTS chat_agent_context (
                    chat_id          Utf8 NOT NULL,
                    user_id          Utf8 NOT NULL,
                    last_tool        Utf8?,
                    last_window_from Utf8?,
                    last_window_to   Utf8?,
                    last_date_field  Utf8?,
                    last_project     Utf8?,
                    created_at       Timestamp NOT NULL,
                    updated_at       Timestamp NOT NULL,
                    PRIMARY KEY (chat_id)
                );
                """
            )
        except ydb.Error as error:
            log_table_creation("chat_agent_context", "error")
            log_error(f"Error: {error}", indent=3)
            raise
        log_table_creation("chat_agent_context", "success")
