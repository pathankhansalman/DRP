"""DRP linter with extensible, rule-id based checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


Severity = str  # "style" | "best_practice"


@dataclass(frozen=True)
class LintWarning:
    """Represents a lint warning for a DRP record."""

    rule_id: str
    message: str
    field: str
    severity: Severity


RuleFunc = Callable[[dict[str, Any]], list[LintWarning]]
_RULES: list[RuleFunc] = []


def register_rule(func: RuleFunc) -> RuleFunc:
    """Decorator for registering lint rules."""
    _RULES.append(func)
    return func


@register_rule
def rule_record_id_format(record: dict[str, Any]) -> list[LintWarning]:
    rid = record.get("record_id")
    if isinstance(rid, str) and not re.fullmatch(r"drp-\d+", rid):
        return [LintWarning("DRP001", "record_id should follow 'drp-XXX' (digits).", "record_id", "style")]
    return []


@register_rule
def rule_timestamp_for_decision(record: dict[str, Any]) -> list[LintWarning]:
    if "decision" in record and not record.get("timestamp"):
        return [LintWarning("DRP002", "timestamp is recommended for decision records.", "timestamp", "best_practice")]
    return []


@register_rule
def rule_priority_present(record: dict[str, Any]) -> list[LintWarning]:
    if "decision" not in record:
        return []
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    has_priority = "priority" in record or "priority" in metadata
    is_root = not record.get("parent_record_ids")
    if is_root and not has_priority:
        return [LintWarning("DRP003", "priority is recommended for root decisions.", "priority", "best_practice")]
    return []


@register_rule
def rule_causal_trace_links(record: dict[str, Any]) -> list[LintWarning]:
    trace = record.get("causal_trace")
    if not isinstance(trace, dict):
        return []
    trigger_ids = trace.get("trigger_event_ids")
    related = trace.get("related_decisions")
    has_trigger = isinstance(trigger_ids, list) and len(trigger_ids) > 0
    has_related = isinstance(related, list) and len(related) > 0
    if not has_trigger and not has_related:
        return [LintWarning("DRP004", "causal_trace should include trigger_event_ids or related_decisions.", "causal_trace", "best_practice")]
    return []


@register_rule
def rule_non_empty_description(record: dict[str, Any]) -> list[LintWarning]:
    content = record.get("content")
    if not isinstance(content, dict):
        return []
    description = content.get("description")
    if isinstance(description, str) and description.strip() == "":
        return [LintWarning("DRP005", "content.description must not be empty.", "content.description", "style")]
    return []


def lint_record(record: dict[str, Any]) -> list[LintWarning]:
    """Run all registered lint rules for one DRP record."""
    warnings: list[LintWarning] = []
    for rule in _RULES:
        warnings.extend(rule(record))
    return warnings


def lint_data(data: Any) -> list[LintWarning]:
    """Lint a parsed DRP object or list of DRP objects."""
    if isinstance(data, dict):
        return lint_record(data)
    if isinstance(data, list):
        warnings: list[LintWarning] = []
        for record in data:
            if isinstance(record, dict):
                warnings.extend(lint_record(record))
        return warnings
    return []
