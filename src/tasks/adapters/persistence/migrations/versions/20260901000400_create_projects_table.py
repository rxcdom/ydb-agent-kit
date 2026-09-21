"""Creates the ``projects`` table.

A project is read either by its id or as part of "all projects of this owner".
The second read filters on ``user_id``, which is not the primary key, so it
gets a secondary index; the repository names that index in its query.
"""
from __future__ import annotations

import ydb

from src.shared.infrastructure.database.migration.base import Migration
from src.shared.infrastructure.database.migration.logging import log_error, log_table_creation

CREATE_PROJECTS_TABLE = """
CREATE TABLE IF NOT EXISTS projects (
    project_id  Utf8 NOT NULL,
    user_id     Utf8 NOT NULL,
    name        Utf8 NOT NULL,
    description Utf8?,
    created_at  Timestamp NOT NULL,
    updated_at  Timestamp NOT NULL,
    PRIMARY KEY (project_id),
    INDEX idx_projects_user_id GLOBAL ON (user_id)
);
"""


class CreateProjectsTableMigration(Migration):
    version = "20260901000400"
    description = "Create projects table"

    def get_artifacts(self):
        return {
            "tables": ["projects"],
            "indexes": [("projects", "idx_projects_user_id")],
            "columns": [
                ("projects", "project_id", "Utf8"),
                ("projects", "user_id", "Utf8"),
                ("projects", "name", "Utf8"),
                ("projects", "description", "Utf8?"),
                ("projects", "created_at", "Timestamp"),
                ("projects", "updated_at", "Timestamp"),
            ],
        }

    async def up(self, pool: ydb.aio.QuerySessionPool) -> None:
        log_table_creation("projects", "creating")
        try:
            await pool.execute_with_retries(CREATE_PROJECTS_TABLE)
        except ydb.Error as error:
            log_table_creation("projects", "error")
            log_error(f"Error: {error}", indent=3)
            raise
        log_table_creation("projects", "success")
