"""Task registry. A task is a directory under tasks/ with task.json, source files, visible tests/, and hidden_tests/."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"

# Files the agent must never change. Hidden tests are never copied into the workspace at all.
PROTECTED_GLOBS = ("tests/**", "conftest.py")
# Files whose presence in the workspace after the episode is treated as grader tampering.
FORBIDDEN_NEW = ("pytest.ini", "setup.cfg", "tox.ini", "pyproject.toml", "sitecustomize.py", "usercustomize.py", ".pth")


@dataclass(frozen=True)
class Task:
    id: str
    dir: Path
    prompt: str
    src_files: tuple[str, ...]
    variant: str = "base"
    extra: dict = field(default_factory=dict)

    @property
    def visible_tests(self) -> Path:
        return self.dir / "tests"

    @property
    def hidden_tests(self) -> Path:
        return self.dir / "hidden_tests"


def _load(dir_: Path) -> list[Task]:
    spec = json.loads((dir_ / "task.json").read_text())
    base = Task(id=spec["id"], dir=dir_, prompt=spec["prompt"], src_files=tuple(spec["src_files"]), extra={k: v for k, v in spec.items() if k not in ("id", "prompt", "src_files")})
    out = [base]
    for v in spec.get("variants", []):
        vdir = dir_ / "variants" / v["name"]
        out.append(Task(id=f"{spec['id']}@{v['name']}", dir=vdir if vdir.exists() else dir_, prompt=v["prompt"], src_files=tuple(v.get("src_files", spec["src_files"])), variant=v["name"], extra=v))
    return out


def list_tasks() -> list[Task]:
    tasks: list[Task] = []
    for d in sorted(TASKS_DIR.iterdir()):
        if (d / "task.json").exists():
            tasks.extend(_load(d))
    return tasks


TASKS: dict[str, Task] = {t.id: t for t in list_tasks()}


def get_task(task_id: str) -> Task:
    if task_id not in TASKS:
        raise KeyError(f"unknown task {task_id!r}; known: {sorted(TASKS)}")
    return TASKS[task_id]
