"""OpenEnv-style environment: reset(task_id) -> Observation; step(action) -> StepResult; state().

An action is either a tool call {"tool": name, "args": {...}} or a final message {"final": "summary"}.
The step verifier mirrors K2 Cascade's rule checks: a step must act or claim completion; narrated intent
("I'll run the tests") with no tool call is rejected and logged, never treated as completion.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .grader import Grade, grade_workspace
from .tasks import Task, get_task
from .workspace import REQUIRED, TOOL_NAMES, TOOLS, Workspace

SYSTEM = ("You are a coding agent working inside a project directory. Use the tools to inspect and change files and to run commands. "
          "Always use paths relative to the project directory. Take one small step at a time. When the task is fully done and verified, "
          "reply with a short summary and no tool call.")
NARRATED = ("i'll", "i will", "let me", "first", "next", "now i")
DONE_WORDS = ("done", "pass", "complete", "implemented", "fixed", "finished", "summary")


@dataclass
class Observation:
    task_id: str
    prompt: str
    system: str
    tools: list[dict[str, Any]]
    step: int
    last_result: str = ""


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


def verify_action(action: dict[str, Any], history: list[dict[str, Any]]) -> tuple[bool, str]:
    """Cheap rule checks. Returns (ok, reason)."""
    if "final" in action:
        text = str(action.get("final", "")).strip().lower()
        if not text:
            return False, "empty_final"
        if text.startswith(NARRATED) and not any(w in text for w in DONE_WORDS):
            return False, "no_action"  # "I'll start by running the tests" is not a completion claim
        return True, "final"
    if "tool" not in action:
        return False, "no_action"
    name = action["tool"]
    if name not in TOOL_NAMES:
        return False, f"unknown_tool:{name}"
    args = action.get("args")
    if not isinstance(args, dict):
        return False, "args_not_object"
    for k in REQUIRED[name]:
        if k not in args:
            return False, f"missing_arg:{k}"
    # Repeating an identical call is pointless unless something was written in between (re-running the
    # tests after an edit is the normal loop). Compare against calls since the last write_file.
    key = json.dumps({"tool": name, "args": args}, sort_keys=True)
    recent: list[dict[str, Any]] = []
    for h in reversed(history[-3:]):
        if h.get("tool") == "write_file":
            break
        recent.append(h)
    if key in [json.dumps(h, sort_keys=True) for h in recent]:
        return False, "repeat_call"
    return True, "ok"


class CodingTaskEnv:
    def __init__(self, max_steps: int = 30, trace_path: Path | None = None, tool_timeout: int = 60):
        self.max_steps, self.trace_path, self.tool_timeout = max_steps, trace_path, tool_timeout
        self.task: Task | None = None
        self.ws: Workspace | None = None
        self.step_no = 0
        self.history: list[dict[str, Any]] = []
        self.rejections: list[str] = []
        self.episode_id = ""
        self.grade: Grade | None = None
        self._done = False

    # ---- OpenEnv-style API ----
    def reset(self, task_id: str) -> Observation:
        self.close()
        self.task = get_task(task_id)
        self.ws = Workspace(self.task)
        self.step_no, self.history, self.rejections, self.grade, self._done = 0, [], [], None, False
        self.episode_id = uuid.uuid4().hex[:8]
        self._trace({"event": "reset", "task_id": task_id})
        return self._obs("")

    def step(self, action: dict[str, Any]) -> StepResult:
        assert self.ws and self.task, "call reset() first"
        if self._done:
            return StepResult(self._obs("episode is over"), 0.0, True, {"reason": "already_done"})
        self.step_no += 1
        t0 = time.time()
        ok, reason = verify_action(action, self.history)
        info: dict[str, Any] = {"step": self.step_no, "verdict": reason, "ok": ok}
        result, reward, done = "", 0.0, False
        if not ok:
            self.rejections.append(reason)
            result = f"rejected: {reason}"
        elif reason == "final":
            done = True
            self.grade = grade_workspace(self.ws)
            reward = self.grade.reward
            info["grade"] = asdict(self.grade)
            result = f"graded: reward={reward}"
        else:
            self.history.append({"tool": action["tool"], "args": action["args"]})
            result = self.ws.execute(action["tool"], action["args"], timeout=self.tool_timeout)
        if not done and self.step_no >= self.max_steps:
            done = True
            self.grade = grade_workspace(self.ws)
            reward = self.grade.reward
            info["grade"] = asdict(self.grade)
            info["reason"] = "max_steps"
        info["latency_ms"] = int((time.time() - t0) * 1000)
        self._trace({"event": "step", "action": action, "result": result[:2000], "reward": reward, "done": done, **{k: v for k, v in info.items() if k != "grade"}})
        if done:
            self._done = True
            self._trace({"event": "end", "grade": asdict(self.grade) if self.grade else None, "rejections": self.rejections})
        return StepResult(self._obs(result), reward, done, info)

    def state(self) -> dict[str, Any]:
        return {"episode_id": self.episode_id, "task_id": self.task.id if self.task else None, "step": self.step_no,
                "done": self._done, "history": list(self.history), "rejections": list(self.rejections),
                "grade": asdict(self.grade) if self.grade else None, "workspace": str(self.ws.root) if self.ws else None}

    def close(self) -> None:
        if self.ws:
            self.ws.close()
            self.ws = None

    # ---- helpers ----
    def _obs(self, last: str) -> Observation:
        assert self.task
        return Observation(self.task.id, self.task.prompt, SYSTEM, TOOLS, self.step_no, last)

    def _trace(self, rec: dict[str, Any]) -> None:
        if not self.trace_path:
            return
        rec = {"episode_id": self.episode_id, "ts": round(time.time(), 3), **rec}
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
