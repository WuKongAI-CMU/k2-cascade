"""Three tools. Kept deliberately small so step labels are comparable across sizes."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

MAX_OUT = 6000

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "shell",
        "description": "Run a shell command inside the project directory and return stdout+stderr (truncated).",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string", "description": "Command to run with bash -c"}}, "required": ["cmd"]},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file relative to the project directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a UTF-8 text file relative to the project directory with the full new content.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    }},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
REQUIRED = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}


def _trunc(s: str) -> str:
    return s if len(s) <= MAX_OUT else s[: MAX_OUT // 2] + f"\n...[{len(s) - MAX_OUT} chars truncated]...\n" + s[-MAX_OUT // 2:]


def _safe(cwd: Path, rel: str) -> Path:
    p = (cwd / rel).resolve()
    if cwd.resolve() not in p.parents and p != cwd.resolve():
        raise ValueError(f"path escapes project dir: {rel}")
    return p


def execute(name: str, args: dict[str, Any], cwd: Path, timeout: int = 120) -> str:
    try:
        if name == "shell":
            r = subprocess.run(["bash", "-lc", args["cmd"]], cwd=cwd, capture_output=True, text=True, timeout=timeout)
            out = (r.stdout or "") + (r.stderr or "")
            return _trunc(out) + f"\n[exit {r.returncode}]"
        if name == "read_file":
            return _trunc(_safe(cwd, args["path"]).read_text())
        if name == "write_file":
            p = _safe(cwd, args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            return f"wrote {len(args['content'])} chars to {args['path']}"
        return f"error: unknown tool {name}"
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {e}"
