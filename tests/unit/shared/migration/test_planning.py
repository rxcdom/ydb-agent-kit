"""The planner decides which discovered migration files run on the next ``up``.

Four cases exist:

1. no history row: a new migration, queued;
2. ``failed``: the previous run failed, queued again;
3. ``in_progress``: the previous run never reached a terminal status (the
   process died in the middle of ``up()``), queued again, which is safe because
   every migration is idempotent by contract;
4. ``completed``: skipped.

Without case 3 an interrupted migration would stay ``in_progress`` forever:
the planner would skip it and the final completeness check would refuse every
later run until someone edited the history table by hand.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import patch

from src.shared.domain.migration import MigrationRecord
from src.shared.infrastructure.database.migration.discovery import MigrationFile
from src.shared.infrastructure.database.migration.planning import plan_migrations_to_run

_GET_APPLIED = "src.shared.infrastructure.database.migration.planning.get_applied_migrations"


def _file(*, module: str, version: str) -> MigrationFile:
    return MigrationFile(
        path=Path(f"/fake/{module}/{version}_change.py"),
        module_name=module,
        version=version,
    )


def _record(*, module: str, version: str, status: str) -> MigrationRecord:
    return MigrationRecord(
        version=version,
        module_name=module,
        description="",
        applied_at=dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc),
        checksum="abc",
        applied_by="local",
        status=status,
        error_message=None,
    )


async def test_new_migration_is_queued() -> None:
    discovered = [_file(module="tasks", version="20260101000000")]

    with patch(_GET_APPLIED, return_value={}):
        plan = await plan_migrations_to_run(pool=object(), discovered_files=discovered)

    assert [m.version for m in plan] == ["20260101000000"]


async def test_completed_migration_is_skipped() -> None:
    migration_file = _file(module="tasks", version="20260101000000")
    applied = {
        "tasks:20260101000000": _record(
            module="tasks", version="20260101000000", status="completed"
        ),
    }

    with patch(_GET_APPLIED, return_value=applied):
        plan = await plan_migrations_to_run(pool=object(), discovered_files=[migration_file])

    assert plan == []


async def test_failed_migration_is_requeued() -> None:
    migration_file = _file(module="tasks", version="20260101000000")
    applied = {
        "tasks:20260101000000": _record(
            module="tasks", version="20260101000000", status="failed"
        ),
    }

    with patch(_GET_APPLIED, return_value=applied):
        plan = await plan_migrations_to_run(pool=object(), discovered_files=[migration_file])

    assert [m.version for m in plan] == ["20260101000000"]


async def test_in_progress_migration_is_requeued() -> None:
    """A migration left at ``in_progress`` by an interrupted run is run again."""
    migration_file = _file(module="tasks", version="20260521150000")
    applied = {
        "tasks:20260521150000": _record(
            module="tasks", version="20260521150000", status="in_progress"
        ),
    }

    with patch(_GET_APPLIED, return_value=applied):
        plan = await plan_migrations_to_run(pool=object(), discovered_files=[migration_file])

    assert [m.version for m in plan] == ["20260521150000"]


async def test_module_filter_applies_to_in_progress_migrations() -> None:
    tasks_file = _file(module="tasks", version="20260521150000")
    agent_file = _file(module="agent", version="20260522120000")
    applied = {
        "tasks:20260521150000": _record(
            module="tasks", version="20260521150000", status="in_progress"
        ),
        "agent:20260522120000": _record(
            module="agent", version="20260522120000", status="in_progress"
        ),
    }

    with patch(_GET_APPLIED, return_value=applied):
        plan = await plan_migrations_to_run(
            pool=object(),
            discovered_files=[tasks_file, agent_file],
            module_name_filter="tasks",
        )

    assert [m.module_name for m in plan] == ["tasks"]
    assert [m.version for m in plan] == ["20260521150000"]


async def test_mixed_states_queue_new_failed_and_in_progress_in_discovery_order() -> None:
    completed = _file(module="tasks", version="20260101000000")
    failed = _file(module="tasks", version="20260201000000")
    in_progress = _file(module="tasks", version="20260301000000")
    new = _file(module="tasks", version="20260601000000")
    applied = {
        "tasks:20260101000000": _record(
            module="tasks", version="20260101000000", status="completed"
        ),
        "tasks:20260201000000": _record(
            module="tasks", version="20260201000000", status="failed"
        ),
        "tasks:20260301000000": _record(
            module="tasks", version="20260301000000", status="in_progress"
        ),
    }

    with patch(_GET_APPLIED, return_value=applied):
        plan = await plan_migrations_to_run(
            pool=object(),
            discovered_files=[completed, failed, in_progress, new],
        )

    assert [m.version for m in plan] == [
        "20260201000000",
        "20260301000000",
        "20260601000000",
    ]
