"""Run scripted policies against every task and print a table. `python -m k2_tasks_env.cli [--trace traces/x.jsonl]`"""
from __future__ import annotations

import argparse
from pathlib import Path

from .env import CodingTaskEnv
from .policies import HACKS, hack_hardcode, oracle, run_policy
from .tasks import list_tasks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    a = ap.parse_args(argv)
    tasks = [t for t in list_tasks() if not a.tasks or t.id in a.tasks]
    rows = []
    for t in tasks:
        policies = {"oracle": oracle(t.id), **{k: f(t.id) for k, f in HACKS.items()}}
        if t.id.split("@")[0] == "slug-bug":
            policies["hardcode"] = hack_hardcode(t.id)
        for name, pol in policies.items():
            env = CodingTaskEnv(trace_path=a.trace, max_steps=8)
            res = run_policy(env, t.id, pol, max_steps=8)
            g = res.info.get("grade", {})
            rows.append((t.id, name, res.reward, f"{g.get('visible_pass')}/{g.get('visible_total')}", f"{g.get('hidden_pass')}/{g.get('hidden_total')}", "ok" if g.get("integrity_ok") else ";".join(g.get("violations", [])) or "-", ",".join(env.state()["rejections"]) or "-"))
            env.close()
    w = [max(len(str(r[i])) for r in rows + [("task", "policy", "reward", "visible", "hidden", "integrity", "rejections")]) for i in range(7)]
    hdr = ("task", "policy", "reward", "visible", "hidden", "integrity", "rejections")
    print(" | ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)))
    print("-+-".join("-" * x for x in w))
    for r in rows:
        print(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


if __name__ == "__main__":
    main()
