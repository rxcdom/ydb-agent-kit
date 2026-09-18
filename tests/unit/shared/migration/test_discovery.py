from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.shared.infrastructure.database.migration.base import Migration
from src.shared.infrastructure.database.migration.discovery import (
    MigrationFile,
    compute_checksum,
    discover_migration_files,
)

_MIGRATION_SOURCE = '''
from src.shared.infrastructure.database.migration.base import Migration


class ScratchMigration(Migration):
    version = "{version}"
    description = "{description}"

    async def up(self, pool):
        return None
'''


def _versions_dir(src_root: Path, module: str) -> Path:
    directory = src_root / module / "adapters" / "persistence" / "migrations" / "versions"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_migration(src_root: Path, module: str, version: str, summary: str) -> Path:
    path = _versions_dir(src_root, module) / f"{version}_{summary}.py"
    path.write_text(
        _MIGRATION_SOURCE.format(version=version, description=summary), encoding="utf-8"
    )
    return path


@pytest.fixture
def src_root(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    root.mkdir()
    return root


def test_files_of_all_modules_are_ordered_globally_by_version(src_root: Path) -> None:
    _write_migration(src_root, "orders", "20260301000000", "create_orders")
    _write_migration(src_root, "billing", "20260101000000", "create_invoices")
    _write_migration(src_root, "orders", "20260201000000", "create_customers")

    files = discover_migration_files(src_root)

    assert [(f.module_name, f.version) for f in files] == [
        ("billing", "20260101000000"),
        ("orders", "20260201000000"),
        ("orders", "20260301000000"),
    ]
    assert all(f.path.is_absolute() for f in files)


def test_files_that_are_not_migrations_are_skipped(src_root: Path, capsys) -> None:
    kept = _write_migration(src_root, "orders", "20260101000000", "create_orders")
    versions = _versions_dir(src_root, "orders")
    (versions / "__init__.py").write_text("", encoding="utf-8")
    (versions / "helpers.py").write_text("VALUE = 1\n", encoding="utf-8")
    (versions / "notes.txt").write_text("not python\n", encoding="utf-8")
    (versions / "2026_short_version.py").write_text("VALUE = 2\n", encoding="utf-8")
    outside = src_root / "orders" / "adapters" / "20260101000001_misplaced.py"
    outside.write_text("VALUE = 3\n", encoding="utf-8")

    files = discover_migration_files(src_root)

    assert [f.path for f in files] == [kept.resolve()]
    # A digit-led name that misses the convention is reported, not hidden.
    assert "2026_short_version.py" in capsys.readouterr().out


def test_duplicate_version_across_modules_is_rejected(src_root: Path) -> None:
    first = _write_migration(src_root, "billing", "20260101000000", "create_invoices")
    second = _write_migration(src_root, "orders", "20260101000000", "create_orders")

    with pytest.raises(ValueError, match="Duplicate migration version found") as raised:
        discover_migration_files(src_root)

    assert str(first.resolve()) in str(raised.value)
    assert str(second.resolve()) in str(raised.value)


def test_empty_tree_yields_no_files(src_root: Path) -> None:
    assert discover_migration_files(src_root) == []


def test_missing_source_root_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_migration_files(tmp_path / "absent")


def test_import_class_loads_a_tree_outside_the_import_path(src_root: Path) -> None:
    _write_migration(src_root, "orders", "20260101000000", "create_orders")
    (migration_file,) = discover_migration_files(src_root)

    migration_cls = migration_file.import_class()

    assert issubclass(migration_cls, Migration)
    assert migration_cls.version == "20260101000000"
    assert migration_cls.description == "create_orders"


def test_import_class_rejects_a_file_whose_class_version_differs(src_root: Path) -> None:
    path = _write_migration(src_root, "orders", "20260101000000", "create_orders")
    renamed = path.with_name("20260202000000_create_orders.py")
    path.rename(renamed)
    (migration_file,) = discover_migration_files(src_root)

    with pytest.raises(ImportError, match="20260202000000"):
        migration_file.import_class()


def test_checksum_is_the_digest_of_the_file_content(src_root: Path) -> None:
    path = _write_migration(src_root, "orders", "20260101000000", "create_orders")
    migration_file = MigrationFile(path=path, module_name="orders", version="20260101000000")

    before = compute_checksum(migration_file)
    assert before == hashlib.sha256(path.read_bytes()).hexdigest()

    path.write_text(path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    assert compute_checksum(migration_file) != before
