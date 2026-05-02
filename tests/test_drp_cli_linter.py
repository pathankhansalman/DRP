"""Basic tests for the new CLI and linter modules."""

from drp_cli import _expand_paths
from linter import lint_record


def test_expand_paths_with_single_file(tmp_path):
    file_path = tmp_path / "record.json"
    file_path.write_text("{}", encoding="utf-8")
    expanded = _expand_paths([str(file_path)])
    assert str(file_path) in expanded


def test_lint_record_warns_on_bad_record_id():
    warnings = lint_record({"record_id": "wrong-id", "decision": "x"})
    assert any(w.field == "record_id" for w in warnings)
