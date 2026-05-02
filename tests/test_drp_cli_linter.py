"""Tests for enhanced CLI and linter modules."""

import json

from drp_cli import EXIT_LINT_FAIL, EXIT_OK, _expand_paths, main
from linter import lint_record


def test_expand_paths_with_single_file(tmp_path):
    file_path = tmp_path / "record.json"
    file_path.write_text("{}", encoding="utf-8")
    expanded = _expand_paths([str(file_path)])
    assert str(file_path) in expanded


def test_lint_record_warns_on_bad_record_id_with_rule_id():
    warnings = lint_record({"record_id": "wrong-id", "decision": "x"})
    assert any(w.rule_id == "DRP001" for w in warnings)


def test_cli_validate_json_mode_examples():
    rc = main(["validate", "examples/minimal_valid.json", "--format", "json"])
    assert rc == EXIT_OK


def test_cli_lint_fail_on_warn(tmp_path, capsys):
    data = {"record_id": "bad", "decision": "x"}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    rc = main(["lint", str(path), "--fail-on-warn"])
    _ = capsys.readouterr()
    assert rc == EXIT_LINT_FAIL
