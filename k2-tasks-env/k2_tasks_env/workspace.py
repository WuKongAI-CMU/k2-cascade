"""A workspace is a fresh copy of a task's visible files in a temp dir. Tools are confined to it."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .tasks import PROTECTED_GLOBS, Task

MAX_OUT = 6000
TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "shell", "description": "Run a shell command inside the project directory and return stdout+stderr (truncated).",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file relative to the project directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a UTF-8 text file relative to the project directory with the full new content.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
REQUIRED = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}


def _trunc(s: str) -> str:
    return s if len(s) <= MAX_OUT else s[: MAX_OUT // 2] + f"\n...[{len(s) - MAX_OUT} chars truncated]...\n" + s[-MAX_OUT // 2:]


def file_hashes(root: Path, globs=PROTECTED_GLOBS) -> dict[str, str]:
    out: dict[str, str] = {}
    for g in globs:
        for p in sorted(root.glob(g)):
            if p.is_file():
                out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class Workspace:
    def __init__(self, task: Task, root: Path | None = None):
        self.task = task
        self._tmp = None if root else tempfile.TemporaryDirectory(prefix=f"k2env-{task.id.replace('@', '-')}-")
        self.root = Path(root or self._tmp.name).resolve()
        # copy everything the agent may see: source files, visible tests, conftest. Never hidden_tests, never variants.
        for item in task.dir.iterdir():
            if item.name in ("hidden_tests", "variants", "task.json", "__pycache__", ".pytest_cache"):
                continue
            dst = self.root / item.name
            if item.is_dir():
                shutil.copytree(item, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
            else:
                shutil.copy2(item, dst)
        self.initial_hashes = file_hashes(self.root)
        self.initial_files = {str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()}

    def close(self) -> None:
        if self._tmp:
            self._tmp.cleanup()

    # ---- path confinement ----
    def _safe(self, rel: str) -> Path:
        for prefix in ("/workspace/", "/project/", "/task/", str(self.root) + "/"):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
                break
        p = (self.root / rel).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"path escapes project dir: {rel}")
        return p

    # ---- tools ----
    def execute(self, name: str, args: dict[str, Any], timeout: int = 60) -> str:
        try:
            if name == "shell":
                # `python` / `pytest` inside the workspace must resolve to the interpreter running the env,
                # otherwise an agent on a machine whose system python lacks pytest cannot run the tests.
                env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(self.root),
                       "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")}
                r = subprocess.run(["bash", "-lc", str(args["cmd"])], cwd=self.root, capture_output=True, text=True, timeout=timeout, env=env)
                return _trunc((r.stdout or "") + (r.stderr or "")) + f"\n[exit {r.returncode}]"
            if name == "read_file":
                return _trunc(self._safe(str(args["path"])).read_text())
            if name == "write_file":
                p = self._safe(str(args["path"]))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(args["content"]))
                return f"wrote {len(str(args['content']))} chars to {args['path']}"
            return f"error: unknown tool {name}"
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {timeout}s"
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"
