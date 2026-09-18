"""Orchestrates a migration run."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.shared.infrastructure.database.ydb.connection import YDBConnection

from .discovery import DEFAULT_SRC_ROOT, MigrationFile, discover_migration_files
from .executor import run_migration
from .logging import (
    ICON_MIGRATION,
    log_error,
    log_info,
    log_section_header,
    log_separator,
    log_step,
    log_success,
)
from .planning import plan_migrations_to_run
from .repository import ensure_migrations_table, migration_key
from .verification import verify_all_migrations_completed


async def apply_pending_migrations(
    connection: YDBConnection,
    applied_by: str,
    module_name: Optional[str] = None,
    src_root: Path = DEFAULT_SRC_ROOT,
) -> List[MigrationFile]:
    """Discover, plan and apply every pending migration.

    Migrations run one at a time in global version order. The first failure
    stops the run and propagates. After the last migration the history is
    checked once more, so the run succeeds only when no discovered migration
    is left failed or unfinished.

    Args:
        connection: the open YDB connection; the caller keeps ownership of it.
        applied_by: who or what runs the migrations; stored with every row.
        module_name: apply the migrations of this module only.
        src_root: the source tree to search for migration files.

    Returns:
        The migrations applied by this call; empty when nothing was pending.
    """
    log_section_header("Database Migration Process", ICON_MIGRATION)
    log_info(f"Applied by: {applied_by}")
    if module_name:
        log_info(f"Module filter: {module_name}")

    log_step(1, 4, "Ensuring the migrations table exists")
    await ensure_migrations_table(connection.pool)

    log_step(2, 4, "Discovering migration files")
    discovered_files = discover_migration_files(src_root)
    log_info(f"Found {len(discovered_files)} migration file(s)")

    log_step(3, 4, "Planning migrations to run")
    pending = await plan_migrations_to_run(connection.pool, discovered_files, module_name)
    if not pending:
        log_success("No pending migrations to apply.")
        return []

    log_step(4, 4, f"Applying {len(pending)} pending migration(s)")
    log_separator()
    for migration_file in pending:
        try:
            await run_migration(connection, migration_file, applied_by)
        except Exception:
            # run_migration has logged the cause and recorded the failed status.
            key = migration_key(migration_file.module_name, migration_file.version)
            log_error(f"Migration {key} failed. Halting the migration process.")
            raise

    log_separator()
    log_info("Checking that every migration has completed...")
    await verify_all_migrations_completed(connection.pool, discovered_files)
    log_success("All pending migrations applied and verified.")
    return pending
