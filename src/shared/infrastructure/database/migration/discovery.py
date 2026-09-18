"""Finds migration files on disk and loads the classes they define.

Convention: ``src/{module}/adapters/persistence/migrations/versions/`` holds
files named ``{version}_{summary}.py`` where ``version`` is a 14-digit
timestamp. Discovery touches the file system only; it never talks to the
database.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .base import Migration
from .logging import log_warning

# .../src/shared/infrastructure/database/migration/discovery.py -> .../src
DEFAULT_SRC_ROOT = Path(__file__).resolve().parents[4]

MIGRATIONS_GLOB = "*/adapters/persistence/migrations/versions/*.py"

_MIGRATION_FILE_PATTERN = re.compile(r"^(?P<version>\d{14})_\w+\.py$")


@dataclass(frozen=True)
class MigrationFile:
    """A migration file found on disk."""

    path: Path
    module_name: str
    version: str

    def import_class(self) -> type[Migration]:
        """Load the file and return the migration class carrying this version.

        The file is loaded from its location rather than through the import
        system, so a migration tree does not have to be importable as a
        package: any ``src_root`` works, and a file name starting with digits
        is not a problem.
        """
        qualified_name = f"discovered_migrations.{self.module_name}.{self.path.stem}"
        spec = importlib.util.spec_from_file_location(qualified_name, self.path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the migration file {self.path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for attribute in vars(module).values():
            if (
                inspect.isclass(attribute)
                and issubclass(attribute, Migration)
                and attribute is not Migration
                and getattr(attribute, "version", None) == self.version
            ):
                return attribute

        raise ImportError(
            f"No Migration class with version {self.version} found in {self.path}"
        )


def compute_checksum(migration_file: MigrationFile) -> str:
    """Return the SHA-256 of the migration file as stored on disk."""
    return hashlib.sha256(migration_file.path.read_bytes()).hexdigest()


def discover_migration_files(src_root: Path = DEFAULT_SRC_ROOT) -> List[MigrationFile]:
    """Find the migration files of every module, ordered globally by version.

    Files that do not follow the naming convention (package markers, helpers)
    are not migrations and are skipped; a file that starts with a digit but
    still does not match is most likely a typo and is reported.

    Raises:
        FileNotFoundError: ``src_root`` is not a directory.
        ValueError: two files share a version.
    """
    root = Path(src_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Migration source root is not a directory: {root}")

    discovered: List[MigrationFile] = []
    for path in root.glob(MIGRATIONS_GLOB):
        match = _MIGRATION_FILE_PATTERN.match(path.name)
        if match is None:
            if path.name[0].isdigit():
                log_warning(
                    f"Skipping {path}: a migration file is named "
                    f"<14-digit version>_<summary>.py",
                    indent=2,
                )
            continue
        discovered.append(
            MigrationFile(
                path=path.resolve(),
                module_name=path.resolve().relative_to(root).parts[0],
                version=match.group("version"),
            )
        )

    discovered.sort(key=lambda migration_file: (migration_file.version, migration_file.path))

    seen: Dict[str, Path] = {}
    for migration_file in discovered:
        if migration_file.version in seen:
            raise ValueError(
                f"Duplicate migration version found: {migration_file.version}\n"
                f"File 1: {seen[migration_file.version]}\n"
                f"File 2: {migration_file.path}"
            )
        seen[migration_file.version] = migration_file.path

    return discovered
