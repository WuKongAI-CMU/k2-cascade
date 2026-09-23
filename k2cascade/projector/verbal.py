"""Attach the sender's verbal handoff to a variants file: its greedy answer and a confidence word from its
semantic entropy (terciles over the items: low / medium / high confidence).

uv run python -m k2cascade.projector.verbal --data data/conf.jsonl --se analysis/conf/se_clean.jsonl --out data/conf_verbal_clean.jsonl
"""
from __future__ import annotations

import json


def attach(rows: list[dict], se_rows: dict[int, dict]) -> list[dict]:
    ses = sorted(r["se"] for r in se_rows.values())
    lo, hi = ses[len(ses) // 3], ses[2 * len(ses) // 3]
    out = []
    for i, ex in enumerate(rows):
        r = se_rows.get(i)
        if r is None:
            continue
        conf = "high" if r["se"] <= lo else ("medium" if r["se"] <= hi else "low")
        out.append({**ex, "sender_answer": r["greedy"], "sender_se": r["se"], "sender_conf": conf})
    return out


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--se", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    se = {r["i"]: r for r in (json.loads(l) for l in open(a.se) if l.strip())}
    out = attach(rows, se)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} items with verbal handoff -> {a.out}")


if __name__ == "__main__":
    main()
