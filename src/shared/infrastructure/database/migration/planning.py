"""Decides which discovered migrations still have to run."""
from __future__ import annotations

from typing import List, Optional

import ydb

from .discovery import MigrationFile
from .logging import log_info
from .repository import (
    STATUS_FAILED,
    STATUS_IN_PROGRESS,
    get_applied_migrations,
    migration_key,
)


async def plan_migrations_to_run(
    pool: ydb.aio.QuerySessionPool,
    discovered_files: List[MigrationFile],
    module_name_filter: Optional[str] = None,
) -> List[MigrationFile]:
    """Compare the files on disk with the recorded history.

    A file is queued when it has no history row, when its last run failed, or
    when its last run never reached a terminal status. Completed migrations
    are skipped. The order of ``discovered_files`` is preserved.
    """
    applied = await get_applied_migrations(pool)

    candidates = discovered_files
    if module_name_filter:
        log_info(f"Filtering for module: {module_name_filter}", indent=2)
        candidates = [
            migration_file
            for migration_file in discovered_files
            if migration_file.module_name == module_name_filter
        ]

    pending: List[MigrationFile] = []
    for migration_file in candidates:
        key = migration_key(migration_file.module_name, migration_file.version)
        record = applied.get(key)

        if record is None:
            pending.append(migration_file)
        elif record.status == STATUS_FAILED:
            log_info(f"Queueing failed migration for re-run: {key}", indent=2)
            pending.append(migration_file)
        elif record.status == STATUS_IN_PROGRESS:
            # The executor stamps "in_progress" before it calls up(). A process
            # that died in the middle of up() leaves the row without a terminal
            # status; skipping it would block every later run at the final
            # completeness check. Migrations are idempotent by contract, so
            # running an interrupted one again from the start is safe.
            log_info(
                f"Queueing interrupted migration for re-run: {key} "
                f"(the previous run did not reach a terminal status)",
                indent=2,
            )
            pending.append(migration_file)

    return pending
