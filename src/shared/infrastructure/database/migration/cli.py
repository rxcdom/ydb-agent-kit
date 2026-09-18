"""Command-line entry point of the migration framework.

Usage:
    python -m src.shared.infrastructure.database.migration.cli status
    python -m src.shared.infrastructure.database.migration.cli up [--module NAME]

The connection is configured from the environment (``YDB_ENDPOINT``,
``YDB_DATABASE`` and the optional connection tuning variables); a ``.env`` file
in the working directory is loaded first. ``MIGRATION_APPLIED_BY`` names who
runs the migrations. The process exits with a non-zero code on any failure.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback
from typing import Dict, List, Optional, Sequence

from dotenv import load_dotenv

from src.shared.domain.migration import MigrationRecord
from src.shared.infrastructure.database.ydb.connection import (
    YDBConnection,
    open_ydb_connection,
)
from src.shared.infrastructure.database.ydb.settings import YDBSettings

from .discovery import MigrationFile, discover_migration_files
from .logging import (
    ICON_CHECK,
    ICON_CROSS,
    ICON_DOT,
    log_error,
    log_info,
    log_section_header,
    log_success,
)
from .manager import apply_pending_migrations
from .repository import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    get_applied_migrations,
    migration_key,
)

APPLIED_BY_VARIABLE = "MIGRATION_APPLIED_BY"
DEFAULT_APPLIED_BY = "local"

STATUS_PENDING = "pending"
_STATUS_ICONS = {STATUS_COMPLETED: ICON_CHECK, STATUS_FAILED: ICON_CROSS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="migrate", description="YDB schema migrations")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show every migration with its recorded status")
    up_parser = commands.add_parser("up", help="apply pending migrations")
    up_parser.add_argument(
        "--module", type=str, default=None, help="apply the migrations of one module only"
    )
    return parser


def resolve_applied_by(environ: Optional[Dict[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    return env.get(APPLIED_BY_VARIABLE, "").strip() or DEFAULT_APPLIED_BY


def render_status_lines(
    discovered_files: List[MigrationFile], applied: Dict[str, MigrationRecord]
) -> List[str]:
    """Describe every migration known from disk or from the history.

    A file without a history row is pending; a history row without a file is
    still listed, because it tells that the file was removed after it ran.
    """
    descriptions: Dict[str, Dict[str, Optional[MigrationRecord]]] = {}
    for migration_file in discovered_files:
        key = migration_key(migration_file.module_name, migration_file.version)
        descriptions.setdefault(migration_file.module_name, {})[migration_file.version] = (
            applied.get(key)
        )
    for record in applied.values():
        descriptions.setdefault(record.module_name, {}).setdefault(record.version, record)

    lines: List[str] = []
    for module_name in sorted(descriptions):
        lines.append(f"Module: {module_name}")
        for version in sorted(descriptions[module_name]):
            record = descriptions[module_name][version]
            if record is None:
                lines.append(f"  {ICON_DOT} {version}: {STATUS_PENDING}")
                continue
            icon = _STATUS_ICONS.get(record.status, ICON_DOT)
            lines.append(f"  {icon} {version}: {record.description}")
            lines.append(
                f"      status: {record.status} | applied: {record.applied_at.isoformat()} "
                f"| by: {record.applied_by}"
            )
            if record.error_message:
                lines.append(f"      error: {record.error_message}")
    return lines


async def cmd_status(connection: YDBConnection) -> None:
    discovered_files = discover_migration_files()
    applied = await get_applied_migrations(connection.pool)

    log_section_header("Migration Status")
    lines = render_status_lines(discovered_files, applied)
    if not lines:
        log_info("No migrations found.")
    for line in lines:
        print(line, flush=True)


async def cmd_up(connection: YDBConnection, module_name: Optional[str]) -> None:
    await apply_pending_migrations(
        connection, applied_by=resolve_applied_by(), module_name=module_name
    )
    log_success("Migration command completed")


async def run_command(args: argparse.Namespace) -> None:
    """Open the connection, run the command and always close the connection."""
    connection = await open_ydb_connection(YDBSettings.from_env())
    try:
        if args.command == "status":
            await cmd_status(connection)
        else:
            await cmd_up(connection, module_name=args.module)
    finally:
        await connection.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    try:
        asyncio.run(run_command(args))
    except Exception as error:
        # Process boundary: report the failure and turn it into an exit code.
        log_error(f"Migration command failed: {type(error).__name__}: {error}")
        log_error(f"Traceback:\n{traceback.format_exc()}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
