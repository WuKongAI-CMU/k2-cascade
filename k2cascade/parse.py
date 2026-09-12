"""Parse raw K2 Horizon completions: thinking block + JSON-format tool calls."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

THINK_RE = re.compile(r"^\s*(?:<ifm\|(think|think_fast|think_faster)>\n?)?(.*?)</ifm\|(?:think|think_fast|think_faster)>", re.S)
CALLS_RE = re.compile(r"<ifm\|tool_calls>(.*?)</ifm\|tool_calls>", re.S)
CALL_RE = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", re.S)
ARG_RE = re.compile(r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*(?:<ifm\|arg_type>(.*?)</ifm\|arg_type>\s*)?<ifm\|arg_value>(.*?)</ifm\|arg_value>", re.S)
STOP_TOKENS = ["<|ifm|im_end|>", "<|ifm|endoftext|>"]


@dataclass
class Parsed:
    thinking: str = ""
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def ok(self) -> bool:
        return not self.parse_errors


def _coerce(v: str, typ: str | None):
    """XML format: scalars are plain text, arrays/objects are JSON literals."""
    v = v.strip("\n")
    if typ in ("string", "str") or typ is None and not (v[:1] in "[{" or v in ("true", "false", "null") or _is_number(v)):
        return v
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _is_number(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def _parse_call(body: str) -> dict[str, Any]:
    if body.startswith("{"):
        obj = json.loads(body)
        if not isinstance(obj, dict) or "name" not in obj:
            raise ValueError("tool call is not an object with 'name'")
        obj.setdefault("arguments", {})
        if isinstance(obj["arguments"], str):
            obj["arguments"] = json.loads(obj["arguments"])
        return obj
    name, _, rest = body.partition("\n")
    name = name.strip()
    if not name or "<" in name:
        raise ValueError("missing function name")
    args = {k.strip(): _coerce(v, (t or "").strip() or None) for k, t, v in ARG_RE.findall(rest)}
    if rest.strip() and not args:
        raise ValueError("no arg_key/arg_value pairs found")
    return {"name": name, "arguments": args}


def parse(raw: str, *, generation_prompt_opened_think: bool = True) -> Parsed:
    """`raw` is the model output AFTER the generation prompt.

    With add_generation_prompt the prompt already ends in `<ifm|think>\n`, so the
    output begins inside the thinking block and the opening tag is absent.
    """
    out = Parsed(raw=raw)
    text = raw
    for s in STOP_TOKENS:
        text = text.split(s)[0]
    m = THINK_RE.match(text)
    if m:
        out.thinking = m.group(2).strip()
        text = text[m.end():]
    elif generation_prompt_opened_think and "</ifm|think" not in text:
        # Model never closed the thinking block: everything is thinking, no action.
        out.thinking = text.strip()
        out.parse_errors.append("unclosed_think")
        return out
    calls_m = CALLS_RE.search(text)
    if calls_m:
        out.content = (text[: calls_m.start()] + text[calls_m.end():]).strip()
        for cm in CALL_RE.finditer(calls_m.group(1)):
            body = cm.group(1).strip()
            try:
                out.tool_calls.append(_parse_call(body))
            except (json.JSONDecodeError, ValueError) as e:
                out.parse_errors.append(f"bad_tool_call: {e}: {body[:120]!r}")
    else:
        out.content = text.strip()
        if "<ifm|tool_call" in text:
            out.parse_errors.append("unclosed_tool_calls")
    return out
