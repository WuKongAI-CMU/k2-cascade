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
    ap.add_argument("--small-url", default="http://127.0.0.1:8081")
    ap.add_argument("--small-name", default="k2-3.7b")
    ap.add_argument("--trace", default=None)
    ap.add_argument("--max-steps", type=int, default=30)
    a = ap.parse_args()

    from .loop import run
    small = large = None
    if a.mode in ("small", "cascade"):
        from .local import LocalK2
        small = LocalK2(a.small_url, name=a.small_name)
    if a.mode in ("large", "cascade"):
        from .cloud import CloudK2
        large = CloudK2()
    trace = Path(a.trace or f"traces/{a.mode}.jsonl")
    s = run(a.task, Path(a.cwd).resolve(), small=small, large=large, mode=a.mode, trace_path=trace, max_steps=a.max_steps)
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
