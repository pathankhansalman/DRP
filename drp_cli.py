"""DRP command-line interface for validation and linting."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Iterable

from linter import lint_data
from tools import drp_validator

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _expand_paths(patterns: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            paths.append(pattern)
    return paths


def _validate_files(paths: list[str], strict: bool = False) -> int:
    del strict  # reserved for future strict-mode behavior in validator
    has_errors = False
    for path in paths:
        try:
            result = drp_validator.validate_file(path)
        except (OSError, json.JSONDecodeError) as exc:
            has_errors = True
            print(f"{RED}{path}: ❌ errors: 1{RESET}")
            print(f"  - {exc}")
            continue

        if result.ok:
            print(f"{GREEN}{path}: ✅ valid{RESET}")
        else:
            has_errors = True
            print(f"{RED}{path}: ❌ errors: {len(result.errors)}{RESET}")
            for err in result.errors:
                print(f"  - {err.format()}")
    return 1 if has_errors else 0


def _lint_files(paths: list[str]) -> int:
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{YELLOW}{path}: lint skipped ({exc}){RESET}")
            continue

        warnings = lint_data(data)
        if warnings:
            print(f"{YELLOW}{path}: warnings: {len(warnings)}{RESET}")
            for warning in warnings:
                print(
                    f"  - [{warning.severity}] {warning.field}: {warning.message}"
                )
        else:
            print(f"{GREEN}{path}: lint clean{RESET}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] == "lint":
        lint_parser = argparse.ArgumentParser(prog="drp-validate lint")
        lint_parser.add_argument("paths", nargs="+", help="JSON files or glob patterns")
        lint_args = lint_parser.parse_args(argv[1:])
        return _lint_files(_expand_paths(lint_args.paths))

    parser = argparse.ArgumentParser(prog="drp-validate")
    parser.add_argument("paths", nargs="+", help="JSON files or glob patterns")
    parser.add_argument("--strict", action="store_true", help="Enable strict checks")
    parser.add_argument("--lint", action="store_true", help="Run lint after validation")
    args = parser.parse_args(argv)

    file_paths = _expand_paths(args.paths)
    rc = _validate_files(file_paths, strict=args.strict)
    if args.lint:
        _lint_files(file_paths)
    return rc


if __name__ == "__main__":
    sys.exit(main())
