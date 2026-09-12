"""Summarize trace JSONL files into the table the demo ends on.

Usage: python -m k2cascade.compare traces/*.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table


def load(path: Path):
    steps, summary = [], None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("step") == 0 and "summary" in r:
            summary = r["summary"]
        else:
            steps.append(r)
    return steps, summary


def main(paths: list[str]) -> None:
    con = Console()
    t = Table(title="K2 Cascade: same task, three ways")
    for c in ["run", "mode", "done", "steps", "escalated", "small ok%", "tokens small", "tokens 375B", "sec small", "sec 375B"]:
        t.add_column(c)
    reasons: dict[str, Counter] = defaultdict(Counter)
    for p in paths:
        steps, s = load(Path(p))
        if not s:
            continue
        small_attempts = [r for r in steps if not r.get("escalated") and r["model"] != "k2-375b"]
        small_ok = sum(1 for r in small_attempts if r["ok"])
        pct = f"{100 * small_ok / len(small_attempts):.0f}%" if small_attempts else "-"
        tok = s["tokens"]; ms = s["ms"]
        t.add_row(s["run_id"], s["mode"], "yes" if s["done"] else "no", str(s["steps"]), str(s["escalations"]), pct,
                  str(sum(v for k, v in tok.items() if k != "k2-375b")), str(tok.get("k2-375b", 0)),
                  f"{sum(v for k, v in ms.items() if k != 'k2-375b') / 1000:.0f}", f"{ms.get('k2-375b', 0) / 1000:.0f}")
        for r in small_attempts:
            if not r["ok"]:
                reasons[s["mode"]][r["reason"]] += 1
    con.print(t)
    for mode, c in reasons.items():
        con.print(f"[bold]{mode}[/bold] small-model failure reasons: " + ", ".join(f"{k} x{v}" for k, v in c.most_common()))


if __name__ == "__main__":
    main(sys.argv[1:])
