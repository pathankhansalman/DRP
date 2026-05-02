"""DRP command-line interface for validation and linting."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from typing import Iterable

from linter import LintWarning, lint_data
from tools import drp_validator

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2
EXIT_LINT_FAIL = 3


def _expand_paths(patterns: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    return paths


def _validate_files(paths: list[str]) -> tuple[int, dict]:
    payload = {"status": "OK", "files": []}
    has_errors = False
    for path in paths:
        try:
            result = drp_validator.validate_file(path)
        except (OSError, json.JSONDecodeError) as exc:
            has_errors = True
            payload["files"].append({"path": path, "ok": False, "errors": [str(exc)]})
            continue

        if result.ok:
            payload["files"].append({"path": path, "ok": True, "errors": []})
        else:
            has_errors = True
            payload["files"].append({
                "path": path,
                "ok": False,
                "errors": [e.to_dict() for e in result.errors],
            })
    if has_errors:
        payload["status"] = "FAIL"
    return (EXIT_INVALID if has_errors else EXIT_OK), payload


def _lint_files(paths: list[str]) -> tuple[int, dict]:
    payload = {"status": "OK", "files": []}
    warning_count = 0
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            warnings = lint_data(data)
        except (OSError, json.JSONDecodeError) as exc:
            payload["files"].append({"path": path, "warnings": [], "skipped": str(exc)})
            continue
        warning_count += len(warnings)
        payload["files"].append({
            "path": path,
            "warnings": [w.__dict__ for w in warnings],
        })
    payload["warning_count"] = warning_count
    return EXIT_OK, payload


def _print_human_validate(payload: dict) -> None:
    for file_result in payload["files"]:
        path = file_result["path"]
        if file_result["ok"]:
            print(f"{GREEN}{path}: ✅ valid{RESET}")
        else:
            print(f"{RED}{path}: ❌ errors: {len(file_result['errors'])}{RESET}")
            for err in file_result["errors"]:
                print(f"  - {err}")


def _print_human_lint(payload: dict) -> None:
    for file_result in payload["files"]:
        path = file_result["path"]
        skipped = file_result.get("skipped")
        if skipped:
            print(f"{YELLOW}{path}: lint skipped ({skipped}){RESET}")
            continue
        warnings: list[LintWarning | dict] = file_result["warnings"]
        if warnings:
            print(f"{YELLOW}{path}: warnings: {len(warnings)}{RESET}")
            for warning in warnings:
                if isinstance(warning, dict):
                    print(f"  - [{warning['rule_id']}] [{warning['severity']}] {warning['field']}: {warning['message']}")
        else:
            print(f"{GREEN}{path}: lint clean{RESET}")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] not in {"validate", "lint", "-h", "--help"}:
        argv = ["validate", *argv]

    parser = argparse.ArgumentParser(prog="drp-validate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate DRP JSON files")
    validate_parser.add_argument("paths", nargs="+", help="JSON files or glob patterns")
    validate_parser.add_argument("--format", choices=["human", "json"], default="human")
    validate_parser.add_argument("--strict", action="store_true", help="Reserved strict mode")
    validate_parser.add_argument("--lint", action="store_true", help="Also run lint checks")
    validate_parser.add_argument("--fail-on-warn", action="store_true", help="Return non-zero when lint warnings exist")

    lint_parser = subparsers.add_parser("lint", help="Lint DRP JSON files")
    lint_parser.add_argument("paths", nargs="+", help="JSON files or glob patterns")
    lint_parser.add_argument("--format", choices=["human", "json"], default="human")
    lint_parser.add_argument("--fail-on-warn", action="store_true", help="Return non-zero when warnings exist")

    args = parser.parse_args(argv)

    paths = _expand_paths(args.paths)

    if args.command == "validate":
        rc, validate_payload = _validate_files(paths)
        if args.format == "json":
            print(json.dumps(validate_payload, ensure_ascii=False))
        else:
            _print_human_validate(validate_payload)

        if args.lint:
            _, lint_payload = _lint_files(paths)
            if args.format == "json":
                print(json.dumps(lint_payload, ensure_ascii=False))
            else:
                _print_human_lint(lint_payload)
            if args.fail_on_warn and lint_payload.get("warning_count", 0) > 0 and rc == EXIT_OK:
                return EXIT_LINT_FAIL
        return rc

    _, lint_payload = _lint_files(paths)
    if args.format == "json":
        print(json.dumps(lint_payload, ensure_ascii=False))
    else:
        _print_human_lint(lint_payload)
    if args.fail_on_warn and lint_payload.get("warning_count", 0) > 0:
        return EXIT_LINT_FAIL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
