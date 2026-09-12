"""Render K2 Horizon chat prompts with the official HF chat template.

We render the prompt ourselves (instead of trusting llama-server's --jinja)
so the exact same bytes go to every model size and we control the
tool_call_format. The HF template uses `{% generation %}` blocks that
standard Jinja2 does not know; they are stripped before compilation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment

BOS = "<|ifm|begin_of_text|>"
TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "k2_chat_template.jinja"


def _raise(msg: str):
    raise ValueError(msg)


def _load_template():
    src = TEMPLATE_PATH.read_text()
    src = re.sub(r"\{%-?\s*generation\s*-?%\}", "", src)
    src = re.sub(r"\{%-?\s*endgeneration\s*-?%\}", "", src)
    env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    env.globals["raise_exception"] = _raise
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)
    return env.from_string(src)


_TEMPLATE = None


def render(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    reasoning_effort: str = "low",
    tool_call_format: str = "json",
    add_generation_prompt: bool = True,
) -> str:
    """Return the full prompt string (including BOS) for llama-server /completion."""
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = _load_template()
    # The template refuses assistant turns without a thinking field.
    msgs = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and not any(
            k in m for k in ("think", "think_fast", "think_faster", "reasoning", "reasoning_content")
        ):
            m["reasoning_content"] = ""
        msgs.append(m)
    return _TEMPLATE.render(
        bos_token=BOS,
        messages=msgs,
        tools=tools or [],
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=reasoning_effort,
        tool_call_format=tool_call_format,
        tool_presentation_format="markdown",
    )
