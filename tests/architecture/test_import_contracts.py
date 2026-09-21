"""Runs the import-linter contracts in-process so a plain ``pytest`` enforces them.

The library's use case is called directly instead of its command-line wrapper: the
wrapper reconfigures logging for the whole process and thereby disables every logger
that already exists, which would silence the log assertions of the tests that run
after this one.
"""
from pathlib import Path

from importlinter import configuration
from importlinter.application import use_cases

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_import_contracts_are_kept(monkeypatch):
    monkeypatch.chdir(REPOSITORY_ROOT)
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT))
    configuration.configure()

    passed = use_cases.lint_imports(
        config_filename=str(REPOSITORY_ROOT / ".importlinter"),
        cache_dir=None,
        no_logo=True,
    )

    assert passed, "an import contract is broken; run `lint-imports` for the details"
