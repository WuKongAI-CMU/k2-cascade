"""Cheap rule checks on a proposed step. Returns (ok, reason)."""
from __future__ import annotations

import json
from typing import Any

from .parse import Parsed
from .tools import REQUIRED, TOOL_NAMES


def check(step: Parsed, history_calls: list[dict[str, Any]]) -> tuple[bool, str]:
    if not step.ok:
        return False, "parse:" + step.parse_errors[0].split(":")[0]
    if not step.tool_calls:
        return True, "final"
    if len(step.tool_calls) > 3:
        return False, "too_many_calls"
    for c in step.tool_calls:
        if c["name"] not in TOOL_NAMES:
            return False, f"unknown_tool:{c['name']}"
        if not isinstance(c["arguments"], dict):
            return False, "args_not_object"
        missing = [k for k in REQUIRED[c["name"]] if k not in c["arguments"]]
        if missing:
            return False, f"missing_arg:{missing[0]}"
    key = json.dumps(step.tool_calls, sort_keys=True)
    recent = [json.dumps(h, sort_keys=True) for h in history_calls[-3:]]
    if key in recent:
        return False, "repeat_call"
    return True, "ok"
