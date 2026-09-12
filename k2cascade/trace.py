"""One JSONL line per attempted step. This file IS the dataset."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Trace:
    def __init__(self, path: Path, run: dict[str, Any]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.t0 = time.time()

    def write(self, **rec: Any) -> None:
        rec = {"ts": round(time.time() - self.t0, 3), **self.run, **rec}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
