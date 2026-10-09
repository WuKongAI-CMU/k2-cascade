"""Prime Intellect `verifiers` adapter: `load_environment()` returns a StatefulToolEnv over the same tasks and grader.

Install: pip install verifiers datasets. Evaluate: vf-eval k2_tasks_env -m <model> (after registering the package).
The hidden workspace path is injected into every tool call from state, never shown to the model.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .grader import grade_workspace
from .tasks import list_tasks
from .workspace import Workspace

_WS: dict[str, Workspace] = {}  # workspace registry keyed by rollout id; verifiers state is JSON-ish so we keep objects here


def _ws(key: str, task_id: str) -> Workspace:
    if key not in _WS:
        _WS[key] = Workspace(next(t for t in list_tasks() if t.id == task_id))
    return _WS[key]


def shell(cmd: str, workspace: str = "") -> str:
    """Run a shell command inside the project directory and return stdout+stderr (truncated)."""
    return _WS[workspace].execute("shell", {"cmd": cmd})


def read_file(path: str, workspace: str = "") -> str:
    """Read a UTF-8 text file relative to the project directory."""
    return _WS[workspace].execute("read_file", {"path": path})


def write_file(path: str, content: str, workspace: str = "") -> str:
    """Create or overwrite a UTF-8 text file relative to the project directory with the full new content."""
    return _WS[workspace].execute("write_file", {"path": path, "content": content})


def load_environment(max_turns: int = 30, **kwargs):
    import verifiers as vf
    from datasets import Dataset

    from .env import SYSTEM

    rows = [{"question": t.prompt, "answer": "", "info": {"task_id": t.id}, "task": "k2-tasks"} for t in list_tasks()]
    dataset = Dataset.from_list(rows)

    class K2TasksEnv(vf.StatefulToolEnv):
        async def setup_state(self, state, **kw):
            state = await super().setup_state(state, **kw)
            key = str(id(state))
            _ws(key, state["info"]["task_id"])
            state["workspace_key"] = key
            return state

        def update_tool_args(self, tool_name, tool_args, messages, state, **kw):
            tool_args["workspace"] = state["workspace_key"]
            return tool_args

    async def hidden_test_reward(state, **kw) -> float:
        ws = _WS.get(state.get("workspace_key", ""))
        if ws is None:
            return 0.0
        g = grade_workspace(ws)
        state["grade"] = asdict(g)
        ws.close()
        _WS.pop(state["workspace_key"], None)
        return g.reward

    env = K2TasksEnv(tools=[], max_turns=max_turns, dataset=dataset, system_prompt=SYSTEM,
                     rubric=vf.Rubric(funcs=[hidden_test_reward], weights=[1.0]), **kwargs)
    for fn in (shell, read_file, write_file):
        env.add_tool(fn, args_to_skip=["workspace"])
    return env
