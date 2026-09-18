"""Guard: every migration file of the project carries a unique version.

Discovery refuses a tree in which two files share a 14-digit version, because
the global order of migrations would be undefined. That refusal would otherwise
surface only when migrations are applied; running the real discovery over the
working tree here moves the failure to the unit-test run. The test holds for
any number of migration files, including none.
"""
from __future__ import annotations

from src.shared.infrastructure.database.migration.discovery import (
    DEFAULT_SRC_ROOT,
    discover_migration_files,
)


def test_migration_versions_are_globally_unique() -> None:
    # A duplicate makes discovery raise ValueError naming both files; letting
    # it propagate shows the same diagnostic the ``up`` command would print.
    files = discover_migration_files(DEFAULT_SRC_ROOT)

    versions = [migration_file.version for migration_file in files]
    assert len(versions) == len(set(versions))
    assert versions == sorted(versions)


def test_every_discovered_file_lives_in_the_project_tree_and_matches_its_version() -> None:
    for migration_file in discover_migration_files(DEFAULT_SRC_ROOT):
        assert migration_file.path.is_relative_to(DEFAULT_SRC_ROOT)
        assert migration_file.path.name.startswith(f"{migration_file.version}_")
        assert migration_file.path.relative_to(DEFAULT_SRC_ROOT).parts[0] == (
            migration_file.module_name
        )


def test_every_discovered_file_defines_a_migration_class_with_its_version() -> None:
    for migration_file in discover_migration_files(DEFAULT_SRC_ROOT):
        migration_cls = migration_file.import_class()

        assert migration_cls.version == migration_file.version
        assert migration_cls.description
