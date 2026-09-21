"""Creates the ``tasks`` table.

Three secondary indexes, one per way the table is read besides the primary key:

- ``idx_tasks_user_created`` serves everything that starts from the owner: the
  full listing, windows on the creation axis, and the owner-scoped aggregates.
- ``idx_tasks_user_due`` serves windows on the due axis. "What is overdue" is a
  range on ``due_at`` inside one owner's rows and would otherwise scan them all.
- ``idx_tasks_project_id`` serves "the tasks of this project".

An index is consulted only by a query that names it, so each repository method
says which one it uses.
"""
from __future__ import annotations

import ydb

from src.shared.infrastructure.database.migration.base import Migration
from src.shared.infrastructure.database.migration.logging import log_error, log_table_creation

CREATE_TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id      Utf8 NOT NULL,
    user_id      Utf8 NOT NULL,
    project_id   Utf8?,
    title        Utf8 NOT NULL,
    notes        Utf8?,
    status       Utf8 NOT NULL,
    priority     Utf8 NOT NULL,
    due_at       Timestamp?,
    completed_at Timestamp?,
    created_at   Timestamp NOT NULL,
    updated_at   Timestamp NOT NULL,
    PRIMARY KEY (task_id),
    INDEX idx_tasks_user_created GLOBAL ON (user_id, created_at),
    INDEX idx_tasks_user_due GLOBAL ON (user_id, due_at),
    INDEX idx_tasks_project_id GLOBAL ON (project_id)
);
"""


class CreateTasksTableMigration(Migration):
    version = "20260901000500"
    description = "Create tasks table"

    def get_artifacts(self):
        return {
            "tables": ["tasks"],
            "indexes": [
                ("tasks", "idx_tasks_user_created"),
                ("tasks", "idx_tasks_user_due"),
                ("tasks", "idx_tasks_project_id"),
            ],
            "columns": [
                ("tasks", "task_id", "Utf8"),
                ("tasks", "user_id", "Utf8"),
                ("tasks", "project_id", "Utf8?"),
                ("tasks", "title", "Utf8"),
                ("tasks", "notes", "Utf8?"),
                ("tasks", "status", "Utf8"),
                ("tasks", "priority", "Utf8"),
                ("tasks", "due_at", "Timestamp?"),
                ("tasks", "completed_at", "Timestamp?"),
                ("tasks", "created_at", "Timestamp"),
                ("tasks", "updated_at", "Timestamp"),
            ],
        }

    async def up(self, pool: ydb.aio.QuerySessionPool) -> None:
        log_table_creation("tasks", "creating")
        try:
            await pool.execute_with_retries(CREATE_TASKS_TABLE)
        except ydb.Error as error:
            log_table_creation("tasks", "error")
            log_error(f"Error: {error}", indent=3)
            raise
        log_table_creation("tasks", "success")
