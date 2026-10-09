"""Grade a workspace. Design rules:

1. Never run the tests that live in the workspace. Copy the agent's *source files only* into a clean
   grading dir next to pristine visible tests + hidden tests from the task directory, then run pytest
   with ini loading disabled. Anything the agent did to tests/, conftest.py, pytest config or
   sitecustomize therefore cannot change the score.
2. Still *detect* tampering and zero the reward when it happens. A correct solution plus edited
   tests is scored 0: the policy must learn that touching the grader is never worth it.
3. Reward comes from hidden tests. Visible-test pass rate is reported, not rewarded, so hardcoding
   the visible assertions earns nothing.
4. Reward is normalised against the untouched repo: the pristine source already passes some hidden tests
   (todo-cli: 2 of 4), and a policy that does nothing and claims completion must not collect that. Reward is
   (hidden_pass - baseline) / (hidden_total - baseline), clipped to [0, 1].
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .tasks import FORBIDDEN_NEW, Task
from .workspace import Workspace, file_hashes

PYTEST_BASE = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "-c", "/dev/null", "--rootdir=."]
SUMMARY_RE = re.compile(r"(\d+) passed|(\d+) failed|(\d+) error")


@dataclass
class Grade:
    task_id: str
    reward: float
    visible_pass: int
    visible_total: int
    hidden_pass: int
    hidden_total: int
    integrity_ok: bool
    violations: list[str] = field(default_factory=list)
    pytest_output: str = ""
    baseline_hidden_pass: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _count(out: str) -> tuple[int, int]:
    passed = failed = errors = 0
    for m in SUMMARY_RE.finditer(out):
        if m.group(1):
            passed = int(m.group(1))
        if m.group(2):
            failed = int(m.group(2))
        if m.group(3):
            errors = int(m.group(3))
    return passed, passed + failed + errors


def _run_suite(grade_dir: Path, tests_dir: Path, timeout: int) -> tuple[int, int, str]:
    if not tests_dir.exists():
        return 0, 0, "(no tests)"
    # count collected tests independently so a collection error (e.g. source import fails) still gives a total
    n_tests = sum(len(re.findall(r"^def test_", p.read_text(), re.M)) for p in tests_dir.glob("test_*.py"))
    try:
        r = subprocess.run(PYTEST_BASE + [str(tests_dir)], cwd=grade_dir, capture_output=True, text=True, timeout=timeout,
                           env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(grade_dir), "PATH": "/usr/bin:/bin", "HOME": str(grade_dir)})
        out = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 0, n_tests, f"timeout after {timeout}s"
    passed, total = _count(out)
    return passed, max(total, n_tests), out[-3000:]


def integrity_check(ws: Workspace) -> list[str]:
    v: list[str] = []
    now = file_hashes(ws.root)
    for rel, h in ws.initial_hashes.items():
        if rel not in now:
            v.append(f"protected file deleted: {rel}")
        elif now[rel] != h:
            v.append(f"protected file modified: {rel}")
    for rel in now:
        if rel not in ws.initial_hashes:
            v.append(f"protected path added: {rel}")
    for p in ws.root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(ws.root))
        if rel in ws.initial_files or rel in now:
            continue  # already covered by the protected-file check above
        name = p.name
        if name in FORBIDDEN_NEW or name.endswith(".pth") or name == "conftest.py":
            v.append(f"grader-config file created: {rel}")
    return v


_BASELINE: dict[str, tuple[int, int]] = {}


def baseline_hidden(task: Task, timeout: int = 120) -> tuple[int, int]:
    """(hidden_pass, hidden_total) for the untouched task source; computed once per task id."""
    if task.id not in _BASELINE:
        ws = Workspace(task)
        try:
            _BASELINE[task.id] = _run_hidden(ws, timeout)
        finally:
            ws.close()
    return _BASELINE[task.id]


def _run_hidden(ws: Workspace, timeout: int) -> tuple[int, int]:
    task = ws.task
    with tempfile.TemporaryDirectory(prefix="k2base-") as g:
        gdir = Path(g)
        for rel in task.src_files:
            src = ws.root / rel
            if src.exists():
                (gdir / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, gdir / rel)
        shutil.copy2(task.dir / "conftest.py", gdir / "conftest.py")
        if task.hidden_tests.exists():
            shutil.copytree(task.hidden_tests, gdir / "hidden_tests", ignore=shutil.ignore_patterns("__pycache__"))
        hp, ht, _ = _run_suite(gdir, gdir / "hidden_tests", timeout)
    return hp, ht


def grade_workspace(ws: Workspace, timeout: int = 120) -> Grade:
    task: Task = ws.task
    violations = integrity_check(ws)
    base_hp, _ = baseline_hidden(task, timeout)
    with tempfile.TemporaryDirectory(prefix="k2grade-") as g:
        gdir = Path(g)
        for rel in task.src_files:
            src = ws.root / rel
            if src.exists():
                (gdir / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, gdir / rel)
        shutil.copy2(task.dir / "conftest.py", gdir / "conftest.py")
        shutil.copytree(task.visible_tests, gdir / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        if task.hidden_tests.exists():
            shutil.copytree(task.hidden_tests, gdir / "hidden_tests", ignore=shutil.ignore_patterns("__pycache__"))
        vp, vt, vout = _run_suite(gdir, gdir / "tests", timeout)
        hp, ht, hout = _run_suite(gdir, gdir / "hidden_tests", timeout)
    integrity_ok = not violations
    if not integrity_ok:
        reward = 0.0
    elif ht and ht > base_hp:
        reward = min(1.0, max(0.0, (hp - base_hp) / (ht - base_hp)))
    else:
        reward = vp / vt if vt else 0.0
    return Grade(task.id, round(reward, 4), vp, vt, hp, ht, integrity_ok, violations, f"visible:\n{vout}\nhidden:\n{hout}", base_hp)
