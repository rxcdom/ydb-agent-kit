"""Creates the ``users`` table.

The primary key is the user id, which is also the bearer credential of the
demo, so resolving a request's principal is a single primary-key read.
"""
from __future__ import annotations

import ydb

from src.shared.infrastructure.database.migration.base import Migration
from src.shared.infrastructure.database.migration.logging import log_error, log_table_creation


class CreateUsersTableMigration(Migration):
    version = "20260901000000"
    description = "Create users table"

    def get_artifacts(self):
        return {
            "tables": ["users"],
            "columns": [
                ("users", "user_id", "Utf8"),
                ("users", "display_name", "Utf8?"),
                ("users", "created_at", "Timestamp"),
            ],
        }

    async def up(self, pool: ydb.aio.QuerySessionPool) -> None:
        log_table_creation("users", "creating")
        try:
            await pool.execute_with_retries(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id      Utf8,
                    display_name Utf8?,
                    created_at   Timestamp,
                    PRIMARY KEY (user_id)
                );
                """
            )
        except ydb.Error as error:
            log_table_creation("users", "error")
            log_error(f"Error: {error}", indent=3)
            raise
        log_table_creation("users", "success")
