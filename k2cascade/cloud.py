"""IFM API client (OpenAI-compatible) for K2-Horizon-375B-A23B."""
from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from .local import Usage
from .parse import Parsed

DEFAULT_MODEL = "IFM/K2-Horizon-375B-A23B"
KEY_FILE = os.path.expanduser("~/.config/ifm/env")


def _key_from_file() -> str | None:
    try:
        for line in open(KEY_FILE):
            line = line.strip()
            if line.startswith("IFM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    except FileNotFoundError:
        return None
    return None


class CloudK2:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = "https://api.ifm.ai/v1", name: str = "k2-375b", timeout: float = 600):
        key = os.environ.get("IFM_API_KEY") or _key_from_file()
        if not key:
            raise RuntimeError("IFM_API_KEY not set and ~/.config/ifm/env has no IFM_API_KEY line")
        self.model, self.name = model, name
        self.client = httpx.Client(base_url=base_url, timeout=timeout, headers={"Authorization": f"Bearer {key}"})

    def step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, max_tokens: int = 4096, temperature: float = 0.6, reasoning_effort: str = "low") -> tuple[Parsed, Usage, str]:
        msgs = []
        for m in messages:
            m = dict(m)
            if m.get("role") == "assistant" and m.get("tool_calls"):
                m["tool_calls"] = [
                    {"id": tc.get("id", f"call_{i}"), "type": "function", "function": {"name": tc["function"]["name"], "arguments": json.dumps(tc["function"]["arguments"])}}
                    for i, tc in enumerate(m["tool_calls"])
                ]
            m.pop("reasoning_content", None)
            msgs.append(m)
        t0 = time.perf_counter()
        r = self.client.post("/chat/completions", json={"model": self.model, "messages": msgs, "tools": tools, "max_tokens": max_tokens, "temperature": temperature})
        r.raise_for_status()
        d = r.json()
        choice = d["choices"][0]["message"]
        out = Parsed(raw=json.dumps(choice, ensure_ascii=False), content=choice.get("content") or "", thinking=choice.get("reasoning_content") or "")
        for tc in choice.get("tool_calls") or []:
            try:
                out.tool_calls.append({"id": tc.get("id"), "name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"] or "{}")})
            except json.JSONDecodeError as e:
                out.parse_errors.append(f"bad_tool_call_json: {e}")
        u = d.get("usage", {})
        usage = Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), int((time.perf_counter() - t0) * 1000))
        return out, usage, json.dumps(msgs)[:20000]
