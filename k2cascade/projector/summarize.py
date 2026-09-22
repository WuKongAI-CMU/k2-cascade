"""Collect every Noma / retention json under one or more analysis dirs into a markdown table.

uv run python -m k2cascade.projector.summarize analysis runs/cloud/*/analysis > analysis/summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def rows(dirs: list[str]):
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            if not isinstance(j, dict):
                continue
            if "max_new" in j:                                                   # passage QA
                yield ("qa", d, f.stem, j)
            elif "project" in j and isinstance(j["project"], dict):              # Noma
                yield ("noma", d, f.stem, j)
            elif "retention" in j and "oracle" in j:                             # continuation retention
                yield ("ret", d, f.stem, j)


def main(argv=None) -> None:
    dirs = argv or sys.argv[1:] or ["analysis"]
    noma, ret, qa = [], [], []
    for kind, d, name, j in rows(dirs):
        {"noma": noma, "ret": ret, "qa": qa}[kind].append((d, name, j))
    print("## Noma binding transfer (accuracy; chance = 1/values)\n")
    print("| run | file | attr | pad | layers | n | none | text | project | derange | follow | transfer |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d, name, j in noma:
        g = lambda k: f"{j[k]['acc']:.3f}" if k in j else "-"
        print(f"| {d} | {name} | {j.get('attr', 'colour')} | {j.get('pad', 0)} | "
              f"{'all' if j.get('layers') is None else f'{min(j['layers'])}-{max(j['layers'])}'} | {j.get('n')} | "
              f"{g('none')} | {g('text')} | {g('project')} | {g('derange')} | "
              f"{j['derange'].get('follow_rate', float('nan')):.3f} | {j.get('content_transfer', float('nan')):.3f} |"
              if "derange" in j else
              f"| {d} | {name} | {j.get('attr', 'colour')} | {j.get('pad', 0)} | all | {j.get('n')} | {g('none')} | {g('text')} | {g('project')} | - | - | - |")
    print("\n## Continuation retention (loss; retention = (none-project)/(none-oracle))\n")
    print("| run | file | seqs | none | oracle | project | derange | retention | retention_derange |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d, name, j in ret:
        print(f"| {d} | {name} | {j.get('eval_seqs', j.get('batches'))} | {j['none']:.3f} | {j['oracle']:.3f} | {j['project']:.3f} | "
              f"{j.get('derange', float('nan')):.3f} | {j['retention']:.3f} | {j.get('retention_derange', float('nan')):.3f} |")
    print("\n## Passage QA (SQuAD; F1 / EM of greedy answers, mean gold log-prob per token)\n")
    print("| run | file | n | arm | f1 | em | logp |")
    print("|---|---|---|---|---|---|---|")
    for d, name, j in qa:
        for a, v in j.items():
            if isinstance(v, dict) and "f1" in v:
                print(f"| {d} | {name} | {j['n']} | {a} | {v['f1']:.3f} | {v['em']:.3f} | {v['logp']:.3f} |")


if __name__ == "__main__":
    main()
