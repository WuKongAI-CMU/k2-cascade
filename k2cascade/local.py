"""Client for llama-server's raw /completion endpoint (IFM llama.cpp fork)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from .parse import STOP_TOKENS, Parsed, parse
from .prompt import render


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class LocalK2:
    def __init__(self, base_url: str = "http://127.0.0.1:8081", name: str = "k2-3.7b", timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.client = httpx.Client(timeout=timeout)

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.6,
        reasoning_effort: str = "low",
    ) -> tuple[Parsed, Usage, str]:
        prompt = render(messages, tools, reasoning_effort=reasoning_effort)
        t0 = time.perf_counter()
        r = self.client.post(
            f"{self.base_url}/completion",
            json={
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "stop": STOP_TOKENS,
                "cache_prompt": True,
            },
        )
        r.raise_for_status()
        d = r.json()
        usage = Usage(
            prompt_tokens=int(d.get("tokens_evaluated", 0)) + int(d.get("tokens_cached", 0)),
            completion_tokens=int(d.get("tokens_predicted", 0)),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        return parse(d["content"]), usage, prompt
