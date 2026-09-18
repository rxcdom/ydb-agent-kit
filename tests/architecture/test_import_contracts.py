"""Runs the import-linter contracts in-process so a plain ``pytest`` enforces them."""
from pathlib import Path

from importlinter import cli

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_import_contracts_are_kept(monkeypatch):
    monkeypatch.chdir(REPOSITORY_ROOT)

    exit_code = cli.lint_imports(
        config_filename=str(REPOSITORY_ROOT / ".importlinter"),
        no_cache=True,
    )

    assert exit_code == cli.EXIT_STATUS_SUCCESS, "an import contract is broken; run `lint-imports`"
