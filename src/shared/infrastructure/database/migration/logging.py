"""Console output of the migration framework.

Migrations run from a terminal or a container start-up command, so the
framework prints a hierarchical, human-readable log: section headers, numbered
steps, indented details. Colours are used only when standard output is a
terminal; piped output stays plain text. Errors go to standard error.
"""
from __future__ import annotations

import sys


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


if _supports_color():
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
else:
    RESET = BOLD = DIM = BLUE = GREEN = YELLOW = RED = CYAN = ""

ICON_MIGRATION = "▶"
ICON_TABLE = "▦"
ICON_CHECK = "✓"
ICON_CROSS = "✗"
ICON_WARNING = "!"
ICON_DOT = "•"

RULE = "━" * 3


def _indentation(indent: int) -> str:
    return "  " * indent


def log_section_header(title: str, icon: str = ICON_MIGRATION) -> None:
    """Print the header of a top-level section."""
    text = f"{icon} {title}" if icon else title
    print(f"\n{BLUE}{BOLD}{text}{RESET}", flush=True)


def log_step(step_num: int, total: int, description: str) -> None:
    """Print a numbered step of a multi-step process."""
    print(f"{CYAN}  [{step_num}/{total}]{RESET} {description}", flush=True)


def log_info(message: str, indent: int = 1) -> None:
    print(f"{DIM}{_indentation(indent)}{ICON_DOT}{RESET} {message}", flush=True)


def log_success(message: str, indent: int = 1) -> None:
    print(f"{GREEN}{_indentation(indent)}{ICON_CHECK} {message}{RESET}", flush=True)


def log_warning(message: str, indent: int = 0) -> None:
    print(f"{YELLOW}{_indentation(indent)}{ICON_WARNING} {message}{RESET}", flush=True)


def log_error(message: str, indent: int = 1) -> None:
    print(
        f"{RED}{_indentation(indent)}{ICON_CROSS} {message}{RESET}",
        file=sys.stderr,
        flush=True,
    )


def log_migration_start(module_name: str, version: str, description: str) -> None:
    print(f"\n{CYAN}{BOLD}{RULE} Migration: {module_name}:{version} {RULE}{RESET}", flush=True)
    print(f"{CYAN}  Description:{RESET} {description}", flush=True)


def log_migration_success(module_name: str, version: str) -> None:
    print(
        f"{GREEN}{BOLD}{RULE} {ICON_CHECK} Migration {module_name}:{version} completed "
        f"{RULE}{RESET}\n",
        flush=True,
    )


def log_migration_error(module_name: str, version: str, error: str) -> None:
    print(
        f"{RED}{BOLD}{RULE} {ICON_CROSS} Migration {module_name}:{version} failed {RULE}{RESET}",
        file=sys.stderr,
        flush=True,
    )
    print(f"{RED}  Error: {error}{RESET}\n", file=sys.stderr, flush=True)


def log_table_creation(table_name: str, status: str = "creating") -> None:
    """Report the progress of a table creation.

    ``status`` is one of ``"creating"``, ``"success"`` and ``"error"``.
    """
    if status == "creating":
        print(f"{BLUE}  {ICON_TABLE} Creating table '{table_name}'...{RESET}", flush=True)
    elif status == "success":
        print(f"{GREEN}    {ICON_CHECK} Table '{table_name}' is ready{RESET}", flush=True)
    elif status == "error":
        print(
            f"{RED}    {ICON_CROSS} Failed to create table '{table_name}'{RESET}",
            file=sys.stderr,
            flush=True,
        )
    else:
        raise ValueError(f"Unknown table creation status: {status!r}")


def log_separator(char: str = "─", length: int = 60) -> None:
    print(f"{DIM}{char * length}{RESET}", flush=True)
