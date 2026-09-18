#!/usr/bin/env python3
"""Pre-flight sanitisation gate.

Walks the repository tree and fails when it finds content that must never be
published: non-English text, private identifiers, key-shaped strings, local
paths, references to private documents, engineering-journal phrasing, or
oversized fixtures.

Two kinds of rules exist:

* Generic rules are defined in this file. They describe shapes ("a Cyrillic
  character", "an absolute home-directory path") and carry no private data.
* Private tokens (names, identifiers, addresses that must not leak) are *not*
  stored here, because a public checker that lists them would publish them.
  They are read from an untracked file, one token per line. When that file is
  absent the private-token check is reported as skipped, never as passed.

The file holding real credentials (``.env``) is excluded from the scan by
design: it is untracked, and a key is exactly what it is supposed to contain.
``.env.example`` is scanned instead.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

EXCLUDED_DIRECTORIES = frozenset(
    {".git", ".venv", "__pycache__", ".pytest_cache", ".import_linter_cache"}
)
# Untracked files that hold secrets or private tokens on purpose.
UNSCANNED_FILES = frozenset({".env", ".sanitisation-denylist"})
PRIVATE_DENYLIST_FILENAME = ".sanitisation-denylist"
# This file defines the patterns below, so it necessarily matches them.
PATTERN_DEFINITION_FILE = "scripts/check_sanitised.py"

MAX_TEST_FILE_BYTES = 64 * 1024

REQUIRED_ENV_VARIABLES = (
    "YDB_ENDPOINT",
    "YDB_DATABASE",
    "YDB_CONNECT_ATTEMPTS",
    "YDB_CONNECT_TIMEOUT_SECONDS",
    "YC_FOLDER_ID",
    "YC_API_KEY",
    "YC_IAM_TOKEN",
    "LLM_MODEL_NAME",
    "LLM_TEMPERATURE",
    "AGENT_MAX_ITERATIONS",
    "AGENT_HISTORY_LIMIT",
    "AGENT_TIMEZONE",
    "MIGRATION_APPLIED_BY",
    "LOG_LEVEL",
    "APP_PORT",
    "PYTHONUNBUFFERED",
)
SECRET_ENV_VARIABLES = ("YC_FOLDER_ID", "YC_API_KEY", "YC_IAM_TOKEN")


@dataclass(frozen=True)
class PatternCheck:
    """A line-level regular-expression rule."""

    code: str
    description: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Violation:
    code: str
    path: str
    line_number: int
    detail: str

    def render(self) -> str:
        location = f"{self.path}:{self.line_number}" if self.line_number else self.path
        return f"[{self.code}] {location}: {self.detail}"


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


CYRILLIC_CHECK = PatternCheck(
    code="A1",
    description="Cyrillic character",
    pattern=re.compile(f"[{chr(0x0400)}-{chr(0x04FF)}]"),
)

GENERIC_CHECKS: tuple[PatternCheck, ...] = (
    PatternCheck("A4", "home-directory path", re.compile(r"/Users/")),
    PatternCheck("A4", "private working directory", re.compile(r"wlocal")),
    PatternCheck(
        "A5",
        "private document reference",
        re.compile(
            r"RULES\.md|FINANCE\.md|AI-MODULE\.md|Acquaintance-Plan|REPORT\+OVERVIEW"
            r"|AGENT-V2-REQUIREMENTS|\bTmp/|AGENT-REFACTOR"
        ),
    ),
    PatternCheck("A6", "journal token", re.compile(r"\bincident\b", re.IGNORECASE)),
    PatternCheck("A6", "journal token", re.compile(r"\bregression 20\d\d\b", re.IGNORECASE)),
    PatternCheck("A6", "journal token", re.compile(r"\bverified live\b", re.IGNORECASE)),
    PatternCheck("A6", "journal token", re.compile(r"\bowner decision\b", re.IGNORECASE)),
    PatternCheck("A6", "journal token", re.compile(r"\bPR #\d+")),
    PatternCheck("A6", "journal token", re.compile(r"\bA/B test", re.IGNORECASE)),
    PatternCheck("A8", "key-shaped string", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    PatternCheck("A8", "key-shaped string", re.compile(r"\bAQVN[A-Za-z0-9_-]{8,}")),
)

HEX_DIGEST_CHECK = PatternCheck(
    code="A8",
    description="40-character hex string",
    pattern=re.compile(r"\b[0-9a-fA-F]{40}\b"),
)


def iter_files(root: Path) -> Iterator[Path]:
    """Yield every regular file under ``root`` outside the excluded directories."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIRECTORIES for part in relative_parts[:-1]):
            continue
        yield path


def load_private_tokens(root: Path) -> list[str] | None:
    """Read the untracked private denylist; ``None`` when it does not exist."""
    denylist = root / PRIVATE_DENYLIST_FILENAME
    if not denylist.is_file():
        return None
    tokens = []
    for raw_line in denylist.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            tokens.append(line.lower())
    return tokens


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def scan_patterns(
    root: Path, files: Iterable[Path], checks: Sequence[PatternCheck]
) -> list[Violation]:
    violations = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        for line_number, line in enumerate(_read_lines(path), start=1):
            for check in checks:
                if check.pattern.search(line):
                    violations.append(
                        Violation(
                            check.code,
                            relative,
                            line_number,
                            f"{check.description}: {line.strip()[:160]}",
                        )
                    )
    return violations


def scan_private_tokens(
    root: Path, files: Iterable[Path], tokens: Sequence[str]
) -> list[Violation]:
    """Case-insensitive substring search for every private token.

    The matched line is not echoed: the report may end up in a public log.
    """
    violations = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        for line_number, line in enumerate(_read_lines(path), start=1):
            lowered = line.lower()
            for position, token in enumerate(tokens, start=1):
                if token in lowered:
                    violations.append(
                        Violation(
                            "A2/A3",
                            relative,
                            line_number,
                            f"private denylist token #{position} found",
                        )
                    )
    return violations


def check_fixtures(root: Path, files: Iterable[Path]) -> list[Violation]:
    violations = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if "real_account" in path.name:
            violations.append(Violation("A7", relative, 0, "fixture named after a real account"))
        if relative.startswith("tests/") and path.stat().st_size > MAX_TEST_FILE_BYTES:
            violations.append(
                Violation("A7", relative, 0, f"test file larger than {MAX_TEST_FILE_BYTES} bytes")
            )
    return violations


def check_env_handling(root: Path) -> list[Violation]:
    violations = []

    example = root / ".env.example"
    if not example.is_file():
        violations.append(Violation("A8", ".env.example", 0, "file is missing"))
    else:
        values = {}
        for line in _read_lines(example):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                name, _, value = stripped.partition("=")
                values[name.strip()] = value.strip()
        for name in REQUIRED_ENV_VARIABLES:
            if name not in values:
                violations.append(Violation("A8", ".env.example", 0, f"variable {name} is missing"))
        for name in SECRET_ENV_VARIABLES:
            if values.get(name):
                violations.append(
                    Violation("A8", ".env.example", 0, f"variable {name} must ship empty")
                )

    for ignore_file in (".gitignore", ".dockerignore"):
        path = root / ignore_file
        entries = {line.strip() for line in _read_lines(path)} if path.is_file() else set()
        if ".env" not in entries:
            violations.append(Violation("A8", ignore_file, 0, "does not list .env"))

    return violations


def _is_hex_scope(relative: str) -> bool:
    return (
        relative.startswith(".env")
        or relative == "docker-compose.yml"
        or relative.startswith("src/")
    )


def run_checks(root: Path, private_tokens: Sequence[str] | None) -> Report:
    report = Report()

    scanned = [path for path in iter_files(root) if path.name not in UNSCANNED_FILES]
    pattern_scope = [
        path for path in scanned if path.relative_to(root).as_posix() != PATTERN_DEFINITION_FILE
    ]
    hex_scope = [path for path in pattern_scope if _is_hex_scope(path.relative_to(root).as_posix())]

    report.violations += scan_patterns(root, scanned, (CYRILLIC_CHECK,))
    report.violations += scan_patterns(root, pattern_scope, GENERIC_CHECKS)
    report.violations += scan_patterns(root, hex_scope, (HEX_DIGEST_CHECK,))
    report.violations += check_fixtures(root, scanned)
    report.violations += check_env_handling(root)

    if private_tokens is None:
        report.skipped.append(
            f"A2/A3 private tokens: {PRIVATE_DENYLIST_FILENAME} not present, nothing to match"
        )
    else:
        report.violations += scan_private_tokens(root, scanned, private_tokens)

    return report


def main(argv: Sequence[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else Path(__file__).resolve().parents[1]
    report = run_checks(root, load_private_tokens(root))

    for note in report.skipped:
        print(f"SKIPPED {note}")
    for violation in report.violations:
        print(violation.render())
    if report.ok:
        print("Sanitisation gate: clean.")
        return 0
    print(f"Sanitisation gate: {len(report.violations)} violation(s).")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
