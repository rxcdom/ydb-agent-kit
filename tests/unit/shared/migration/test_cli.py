from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from src.shared.domain.migration import MigrationRecord
from src.shared.infrastructure.database.migration import cli
from src.shared.infrastructure.database.migration.discovery import MigrationFile


def test_status_command_is_parsed() -> None:
    args = cli.build_parser().parse_args(["status"])

    assert args.command == "status"


def test_up_command_defaults_to_every_module() -> None:
    args = cli.build_parser().parse_args(["up"])

    assert args.command == "up"
    assert args.module is None


def test_up_command_accepts_a_module_filter() -> None:
    args = cli.build_parser().parse_args(["up", "--module", "tasks"])

    assert args.command == "up"
    assert args.module == "tasks"


@pytest.mark.parametrize("argv", [[], ["down"], ["status", "--module", "tasks"], ["up", "--all"]])
def test_unknown_or_incomplete_command_lines_are_rejected(argv, capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(argv)

    assert raised.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_applied_by_comes_from_the_environment_with_a_local_default() -> None:
    assert cli.resolve_applied_by({}) == "local"
    assert cli.resolve_applied_by({"MIGRATION_APPLIED_BY": "  "}) == "local"
    assert cli.resolve_applied_by({"MIGRATION_APPLIED_BY": "release-job"}) == "release-job"


class _FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_runtime(monkeypatch):
    """Keep ``main`` away from the environment file and from the network."""
    connection = _FakeConnection()
    calls = []

    async def open_connection(settings):
        return connection

    async def apply(conn, applied_by, module_name=None):
        calls.append((conn, applied_by, module_name))
        return []

    monkeypatch.setattr(cli, "load_dotenv", lambda: False)
    monkeypatch.setattr(cli, "open_ydb_connection", open_connection)
    monkeypatch.setattr(cli, "apply_pending_migrations", apply)
    monkeypatch.setenv("YDB_ENDPOINT", "grpc://localhost:2136")
    monkeypatch.setenv("MIGRATION_APPLIED_BY", "release-job")
    return connection, calls


def test_up_passes_the_connection_the_actor_and_the_filter(fake_runtime) -> None:
    connection, calls = fake_runtime

    exit_code = cli.main(["up", "--module", "agent"])

    assert exit_code == 0
    assert calls == [(connection, "release-job", "agent")]
    assert connection.closed is True


def test_failure_becomes_a_non_zero_exit_code_and_the_connection_is_closed(
    fake_runtime, monkeypatch, capsys
) -> None:
    connection, _calls = fake_runtime

    async def failing(conn, applied_by, module_name=None):
        raise RuntimeError("migration exploded")

    monkeypatch.setattr(cli, "apply_pending_migrations", failing)

    exit_code = cli.main(["up"])

    assert exit_code == 1
    assert connection.closed is True
    assert "migration exploded" in capsys.readouterr().err


def test_connection_failure_becomes_a_non_zero_exit_code(fake_runtime, monkeypatch) -> None:
    async def unreachable(settings):
        raise ConnectionError("database is unreachable")

    monkeypatch.setattr(cli, "open_ydb_connection", unreachable)

    assert cli.main(["status"]) == 1


def test_status_lines_cover_pending_recorded_and_orphaned_migrations() -> None:
    def record(module: str, version: str, status: str, error=None) -> MigrationRecord:
        return MigrationRecord(
            version=version,
            module_name=module,
            description=f"change {version}",
            applied_at=dt.datetime(2026, 9, 1, 12, 0, tzinfo=dt.timezone.utc),
            checksum="abc",
            applied_by="local",
            status=status,
            error_message=error,
        )

    def migration_file(module: str, version: str) -> MigrationFile:
        return MigrationFile(
            path=Path(f"/fake/{version}_change.py"), module_name=module, version=version
        )

    discovered = [
        migration_file("tasks", "20260101000000"),
        migration_file("tasks", "20260201000000"),
        migration_file("agent", "20260301000000"),
    ]
    applied = {
        "tasks:20260101000000": record("tasks", "20260101000000", "completed"),
        "agent:20260301000000": record("agent", "20260301000000", "failed", "index is missing"),
        "legacy:20250101000000": record("legacy", "20250101000000", "completed"),
    }

    text = "\n".join(cli.render_status_lines(discovered, applied))

    assert text.index("Module: agent") < text.index("Module: legacy") < text.index(
        "Module: tasks"
    )
    assert "20260201000000: pending" in text
    assert "20260101000000: change 20260101000000" in text
    assert "status: failed" in text
    assert "error: index is missing" in text
    assert "20250101000000: change 20250101000000" in text


def test_status_lines_are_empty_when_nothing_is_known() -> None:
    assert cli.render_status_lines([], {}) == []
