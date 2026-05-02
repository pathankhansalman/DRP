"""DRP linter with extensible style and best-practice rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Warning:
    """Represents a lint warning for a DRP record."""

    message: str
    field: str
    severity: str  # "style" | "best_practice"


RuleFunc = Callable[[dict[str, Any]], list[Warning]]
_RULES: list[RuleFunc] = []


def register_rule(func: RuleFunc) -> RuleFunc:
    """Decorator for registering lint rules."""
    _RULES.append(func)
    return func


@register_rule
def rule_record_id_format(record: dict[str, Any]) -> list[Warning]:
    rid = record.get("record_id")
    if isinstance(rid, str) and not re.fullmatch(r"drp-\d+", rid):
        return [
            Warning(
                message="record_id should follow the format 'drp-XXX' (digits).",
                field="record_id",
                severity="style",
            )
        ]
    return []


@register_rule
def rule_timestamp_for_decision(record: dict[str, Any]) -> list[Warning]:
    if "decision" in record and not record.get("timestamp"):
        return [
            Warning(
                message="timestamp is recommended for decision records.",
                field="timestamp",
                severity="best_practice",
            )
        ]
    return []


@register_rule
def rule_priority_present(record: dict[str, Any]) -> list[Warning]:
    if "decision" not in record:
        return []
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    has_priority = "priority" in record or "priority" in metadata
    is_root = not record.get("parent_record_ids")
    if is_root and not has_priority:
        return [
            Warning(
                message="priority is recommended for root decisions.",
                field="priority",
                severity="best_practice",
            )
        ]
    return []


@register_rule
def rule_causal_trace_links(record: dict[str, Any]) -> list[Warning]:
    trace = record.get("causal_trace")
    if trace is None:
        return []
    if not isinstance(trace, dict):
        return []
    trigger_ids = trace.get("trigger_event_ids")
    related = trace.get("related_decisions")
    has_trigger = isinstance(trigger_ids, list) and len(trigger_ids) > 0
    has_related = isinstance(related, list) and len(related) > 0
    if not has_trigger and not has_related:
        return [
            Warning(
                message=(
                    "causal_trace should include trigger_event_ids or related_decisions."
                ),
                field="causal_trace",
                severity="best_practice",
            )
        ]
    return []


@register_rule
def rule_non_empty_description(record: dict[str, Any]) -> list[Warning]:
    content = record.get("content")
    if not isinstance(content, dict):
        return []
    description = content.get("description")
    if isinstance(description, str) and description.strip() == "":
        return [
            Warning(
                message="content.description must not be an empty string.",
                field="content.description",
                severity="style",
            )
        ]
    return []


def lint_record(record: dict[str, Any]) -> list[Warning]:
    """Run all registered lint rules for one DRP record."""
    warnings: list[Warning] = []
    for rule in _RULES:
        warnings.extend(rule(record))
    return warnings


def lint_data(data: Any) -> list[Warning]:
    """Lint a parsed DRP object or list of DRP objects."""
    if isinstance(data, dict):
        return lint_record(data)
    if isinstance(data, list):
        warnings: list[Warning] = []
        for record in data:
            if isinstance(record, dict):
                warnings.extend(lint_record(record))
        return warnings
    return []
