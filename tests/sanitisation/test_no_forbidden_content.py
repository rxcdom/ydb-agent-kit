"""Runs the sanitisation gate over the repository tree and tests the gate itself."""
import re
from pathlib import Path

import pytest

from scripts import check_sanitised as gate

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def clean_tree(tmp_path: Path) -> Path:
    example = "".join(f"{name}=\n" for name in gate.REQUIRED_ENV_VARIABLES)
    _write(tmp_path, ".env.example", example)
    _write(tmp_path, ".gitignore", ".env\n")
    _write(tmp_path, ".dockerignore", ".env\n")
    return tmp_path


def test_repository_tree_is_clean():
    report = gate.run_checks(REPOSITORY_ROOT, gate.load_private_tokens(REPOSITORY_ROOT))

    assert report.ok, "\n".join(violation.render() for violation in report.violations)


def test_private_token_check_runs_when_the_denylist_exists():
    tokens = gate.load_private_tokens(REPOSITORY_ROOT)
    if tokens is None:
        pytest.skip("private denylist is not present in this checkout")

    assert tokens, "the private denylist exists but holds no tokens"


def test_clean_tree_passes(clean_tree: Path):
    assert gate.run_checks(clean_tree, private_tokens=[]).ok


def test_missing_denylist_is_reported_as_skipped_not_passed(clean_tree: Path):
    report = gate.run_checks(clean_tree, private_tokens=None)

    assert report.ok
    assert len(report.skipped) == 1


def test_private_token_is_found_case_insensitively_without_echoing_the_line(clean_tree: Path):
    _write(clean_tree, "src/module.py", "name = 'Some-Private-Marker here'\n")

    report = gate.run_checks(clean_tree, private_tokens=["some-private-marker"])

    assert [violation.code for violation in report.violations] == ["A2/A3"]
    assert "marker" not in report.violations[0].detail.lower()


def test_pattern_scan_reports_path_and_line(clean_tree: Path):
    _write(clean_tree, "docs/notes.txt", "first line\nhas forbidden-marker inside\n")
    check = gate.PatternCheck("T1", "test marker", re.compile(r"\bforbidden-marker\b"))

    violations = gate.scan_patterns(clean_tree, list(gate.iter_files(clean_tree)), (check,))

    assert [(v.path, v.line_number) for v in violations] == [("docs/notes.txt", 2)]


def test_non_english_text_is_rejected(clean_tree: Path):
    word = "".join(chr(code) for code in (0x043F, 0x0440, 0x0438))
    _write(clean_tree, "src/module.py", f"greeting = '{word}'\n")

    report = gate.run_checks(clean_tree, private_tokens=[])

    assert {violation.code for violation in report.violations} == {"A1"}


def test_anchored_journal_pattern_ignores_ordinary_prose(clean_tree: Path):
    _write(clean_tree, "README.md", "A coincidental match is not a finding.\n")

    assert gate.run_checks(clean_tree, private_tokens=[]).ok


def test_word_ending_in_sk_followed_by_dash_is_not_a_key(clean_tree: Path):
    _write(clean_tree, "README.md", "A task-manager-style-domain-with-a-long-hyphenated-name.\n")

    assert gate.run_checks(clean_tree, private_tokens=[]).ok


def test_hex_digest_is_rejected_in_source_but_allowed_in_readme(clean_tree: Path):
    digest = "0123456789abcdef" * 2 + "01234567"
    _write(clean_tree, "README.md", f"commit {digest}\n")
    assert gate.run_checks(clean_tree, private_tokens=[]).ok

    _write(clean_tree, "src/config.py", f"KEY = '{digest}'\n")
    report = gate.run_checks(clean_tree, private_tokens=[])
    assert [(v.code, v.path) for v in report.violations] == [("A8", "src/config.py")]


def test_env_file_is_never_scanned(clean_tree: Path):
    _write(clean_tree, ".env", "YC_API_KEY=" + "AQVN" + "x" * 30 + "\n")

    assert gate.run_checks(clean_tree, private_tokens=[]).ok


def test_filled_secret_in_env_example_is_rejected(clean_tree: Path):
    example = "".join(f"{name}=\n" for name in gate.REQUIRED_ENV_VARIABLES)
    _write(clean_tree, ".env.example", example.replace("YC_API_KEY=", "YC_API_KEY=filled"))

    report = gate.run_checks(clean_tree, private_tokens=[])

    assert [violation.code for violation in report.violations] == ["A8"]


def test_env_missing_from_ignore_files_is_rejected(clean_tree: Path):
    _write(clean_tree, ".dockerignore", ".git\n")

    report = gate.run_checks(clean_tree, private_tokens=[])

    assert [(v.code, v.path) for v in report.violations] == [("A8", ".dockerignore")]


def test_oversized_test_fixture_and_real_account_name_are_rejected(clean_tree: Path):
    _write(clean_tree, "tests/fixtures/big.json", "x" * (gate.MAX_TEST_FILE_BYTES + 1))
    _write(clean_tree, "tests/fixtures/payments_real_account.json", "{}")

    report = gate.run_checks(clean_tree, private_tokens=[])

    assert sorted(violation.code for violation in report.violations) == ["A7", "A7"]
