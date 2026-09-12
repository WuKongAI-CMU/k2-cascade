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

LARGE = "k2-375b"
JUDGE = "k2-375b-judge"


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
    con = Console(width=150)
    t = Table(title="K2 Cascade: same task, one row per run")
    for c in ["file", "mode", "done", "steps", "escal.", "steps by model", "local tok", "375B tok", "judge tok", "local s", "375B s", "judge s"]:
        t.add_column(c)
    reasons: dict[str, Counter] = defaultdict(Counter)
    for p in paths:
        steps, s = load(Path(p))
        if not s:
            continue
        accepted = [r for r in steps if r["ok"]]
        by_model = Counter(r["model"] for r in accepted)
        tok, ms = s["tokens"], s["ms"]
        local_tok = sum(v for k, v in tok.items() if k not in (LARGE, JUDGE))
        local_ms = sum(v for k, v in ms.items() if k not in (LARGE, JUDGE))
        t.add_row(Path(p).stem, s["mode"], "yes" if s["done"] else "no", str(s["steps"]), str(s["escalations"]),
                  ", ".join(f"{k}:{v}" for k, v in sorted(by_model.items())),
                  str(local_tok), str(tok.get(LARGE, 0)), str(tok.get(JUDGE, 0)),
                  f"{local_ms / 1000:.0f}", f"{ms.get(LARGE, 0) / 1000:.0f}", f"{ms.get(JUDGE, 0) / 1000:.0f}")
        for r in steps:
            if not r["ok"] and r["model"] != LARGE:
                reasons[r["model"]][r["reason"]] += 1
    con.print(t)
    for model, c in sorted(reasons.items()):
        con.print(f"[bold]{model}[/bold] rejected attempts: " + ", ".join(f"{k} x{v}" for k, v in c.most_common()))


if __name__ == "__main__":
    main(sys.argv[1:])
