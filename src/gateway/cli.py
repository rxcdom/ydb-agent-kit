"""Command-line entry points of the application.

    python -m src.gateway.cli seed --token <user id> [--anchor-date YYYY-MM-DD]
    python -m src.gateway.cli migrate status
    python -m src.gateway.cli migrate up [--module NAME]

``seed`` fills one user's workspace with the deterministic demo dataset. It goes
through the same container and the same use case as the application, so the
seeded rows obey every domain rule. ``migrate`` forwards to the migration
framework's own command line.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime
from typing import Optional, Sequence

from src.accounts.domain.exceptions import InvalidCredentialsError
from src.gateway.di.container import AppContainer
from src.gateway.settings import Settings
from src.shared.domain.exceptions import DomainError
from src.shared.infrastructure.database.migration import cli as migration_cli

logger = logging.getLogger(__name__)


def _iso_day(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD date: {text!r}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ydb-agent-kit")
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed", help="fill a user's workspace with the demo dataset")
    seed.add_argument("--token", required=True, help="the user id returned by POST /api/v1/users")
    seed.add_argument(
        "--anchor-date",
        type=_iso_day,
        default=None,
        help="the day the dataset is laid out around (default: today in AGENT_TIMEZONE)",
    )

    migrate = commands.add_parser("migrate", help="run the migration command line")
    migrate.add_argument("migration_args", nargs=argparse.REMAINDER)
    return parser


async def seed(settings: Settings, token: str, anchor_date: Optional[date]) -> int:
    container = AppContainer(settings=settings)
    await container.init_resources()
    try:
        # Providers that depend on the asynchronous YDB resource resolve to awaitables.
        resolve_principal = await container.accounts.resolve_principal_use_case()
        seed_demo_workspace = await container.tasks.seed_demo_workspace()

        principal = await resolve_principal.execute(token)
        anchor = anchor_date or datetime.now(container.core.timezone()).date()
        summary = await seed_demo_workspace.execute(principal.user_id, anchor)
    except InvalidCredentialsError:
        print("No user has this token. Create one with POST /api/v1/users first.")
        return 1
    except DomainError as error:
        print(f"Seeding failed: {type(error).__name__}: {error}")
        return 1
    finally:
        await container.shutdown_resources()

    print(
        f"Seeded around {summary.anchor_date.isoformat()}: {summary.projects_count} projects, "
        f"{summary.tasks_count} tasks ({summary.open_count} open, {summary.done_count} done, "
        f"{summary.cancelled_count} cancelled; {summary.overdue_count} overdue)."
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "migrate":
        return migration_cli.main(args.migration_args)

    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(seed(settings, args.token, args.anchor_date))


if __name__ == "__main__":
    sys.exit(main())
