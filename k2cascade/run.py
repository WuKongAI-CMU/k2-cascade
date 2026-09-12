"""CLI: python -m k2cascade.run --task '...' --cwd path --mode cascade"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--mode", choices=["small", "large", "cascade"], default="cascade")
    ap.add_argument("--local", action="append", default=None, help="name=url of a local llama-server, smallest first (repeatable)")
    ap.add_argument("--no-cloud", action="store_true", help="do not put the 375B at the top of the ladder")
    ap.add_argument("--trace", default=None)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--no-judge", action="store_true", help="cascade: skip the 375B yes/no check on small-model steps")
    a = ap.parse_args()

    from .loop import run
    from .local import LocalK2
    locals_ = a.local or ["k2-0.9b=http://127.0.0.1:8082", "k2-3.7b=http://127.0.0.1:8081"]
    ladder = [LocalK2(url, name=name) for name, url in (x.split("=", 1) for x in locals_)]
    if not a.no_cloud:
        from .cloud import CloudK2
        ladder.append(CloudK2())
    trace = Path(a.trace or f"traces/{a.mode}.jsonl")
    print("ladder:", " -> ".join(m.name for m in ladder))
    s = run(a.task, Path(a.cwd).resolve(), ladder=ladder, mode=a.mode, trace_path=trace, max_steps=a.max_steps, judge=not a.no_judge)
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
