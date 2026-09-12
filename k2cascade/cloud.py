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

    @staticmethod
    def _convert(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Loop-internal history -> OpenAI wire format (ids on calls, one tool message per call)."""
        msgs: list[dict[str, Any]] = []
        pending_ids: list[str] = []
        for m in messages:
            m = dict(m)
            if m.get("role") == "assistant":
                m["reasoning_content"] = m.get("reasoning_content") or ""
                if m.get("tool_calls"):
                    pending_ids = [tc.get("id") or f"call_{len(msgs)}_{i}" for i, tc in enumerate(m["tool_calls"])]
                    m["tool_calls"] = [
                        {"id": cid, "type": "function", "function": {"name": tc["function"]["name"], "arguments": json.dumps(tc["function"]["arguments"])}}
                        for cid, tc in zip(pending_ids, m["tool_calls"])
                    ]
                    if not m.get("content"):
                        m["content"] = None
                msgs.append(m)
            elif m.get("role") == "tool":
                outputs = m.get("outputs") or [m.get("content", "")]
                for cid, out in zip(pending_ids or [f"call_{len(msgs)}"], outputs):
                    msgs.append({"role": "tool", "tool_call_id": cid, "content": out})
                pending_ids = []
            else:
                msgs.append({"role": m["role"], "content": m.get("content", "")})
        return msgs

    def judge(self, messages: list[dict[str, Any]], proposed: list[dict[str, Any]], task: str) -> tuple[bool, str, "Usage"]:
        """Ask the large model whether a small model's proposed step is a sensible next step. Returns (yes, raw, usage)."""
        history = self._convert(messages)
        recent = history[-6:]
        summary = "\n".join(
            f"- {m['role']}: " + (json.dumps(m.get('tool_calls')) if m.get('tool_calls') else (m.get('content') or '')[:400].replace("\n", " "))
            for m in recent
        )
        q = (f"Task: {task}\n\nRecent history:\n{summary}\n\nProposed next action by a smaller model:\n{json.dumps(proposed)}\n\n"
             "Is this a useful next step toward completing the task (not redundant, not off-track, not repeating work already done)? "
             "Answer with exactly one word: yes or no.")
        t0 = time.perf_counter()
        r = self.client.post("/chat/completions", json={"model": self.model, "messages": [{"role": "user", "content": q}], "max_tokens": 300, "temperature": 0, "reasoning_effort": "low"})
        if r.status_code >= 400:
            raise RuntimeError(f"IFM API {r.status_code}: {r.text[:300]}")
        d = r.json()
        text = (d["choices"][0]["message"].get("content") or "").strip().lower()
        u = d.get("usage", {})
        usage = Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), int((time.perf_counter() - t0) * 1000))
        words = [w.strip(".,!") for w in text.split()]
        verdict = "yes" if ("yes" in words and "no" not in words) else ("no" if "no" in words else "yes")
        return verdict == "yes", text[:200], usage

    def step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, max_tokens: int = 4096, temperature: float = 0.6, reasoning_effort: str = "low") -> tuple[Parsed, Usage, str]:
        msgs = self._convert(messages)
        t0 = time.perf_counter()
        r = self.client.post("/chat/completions", json={"model": self.model, "messages": msgs, "tools": tools, "max_tokens": max_tokens, "temperature": temperature, "reasoning_effort": reasoning_effort})
        if r.status_code >= 400:
            raise RuntimeError(f"IFM API {r.status_code}: {r.text[:500]}")
        d = r.json()
        choice = d["choices"][0]["message"]
        out = Parsed(raw=json.dumps(choice, ensure_ascii=False), content=choice.get("content") or "", thinking=choice.get("reasoning_content") or choice.get("reasoning") or "")
        for tc in choice.get("tool_calls") or []:
            try:
                out.tool_calls.append({"id": tc.get("id"), "name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"] or "{}")})
            except json.JSONDecodeError as e:
                out.parse_errors.append(f"bad_tool_call_json: {e}")
        u = d.get("usage", {})
        usage = Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), int((time.perf_counter() - t0) * 1000))
        return out, usage, json.dumps(msgs)[:20000]
