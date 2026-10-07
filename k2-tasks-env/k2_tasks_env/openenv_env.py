"""OpenEnv-native adapter (meta-pytorch/OpenEnv, `pip install openenv-core`).

Wraps CodingTaskEnv in OpenEnv's Environment / Action / Observation / State models so the package can be
served with OpenEnv's HTTP + WebSocket server and consumed by its clients:

    from k2_tasks_env.openenv_env import app          # FastAPI app
    uvicorn k2_tasks_env.openenv_env:app --port 8001   # POST /reset {"task_id": "slug-bug"}, POST /step {...}

The in-process API (k2_tasks_env.env.CodingTaskEnv) and the stdlib server (k2_tasks_env.server) stay as they are;
this module only translates between the two vocabularies.
"""
from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server import Action, Environment, Observation, State, create_app

from .env import SYSTEM, CodingTaskEnv
from .tasks import list_tasks
from .workspace import TOOLS

TASK_IDS = [t.id for t in list_tasks()]


class K2Action(Action):
    """Either a tool call (tool + args) or a completion claim (final)."""

    tool: Optional[str] = None
    args: dict[str, Any] = {}
    final: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        if self.final is not None:
            return {"final": self.final}
        return {"tool": self.tool, "args": dict(self.args)}


class K2Observation(Observation):
    task_id: str = ""
    prompt: str = ""
    system: str = SYSTEM
    tools: list[dict[str, Any]] = []
    step: int = 0
    last_result: str = ""
    verdict: str = ""


class K2State(State):
    task_id: Optional[str] = None
    done: bool = False
    rejections: list[str] = []
    grade: Optional[dict[str, Any]] = None


class K2TasksOpenEnv(Environment[K2Action, K2Observation, K2State]):
    """One CodingTaskEnv per session; the workspace is a fresh temp dir, so concurrent sessions are safe."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, max_steps: int = 30, trace_path=None):
        super().__init__()
        self._env = CodingTaskEnv(max_steps=max_steps, trace_path=trace_path)
        self._last = K2Observation()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_id: Optional[str] = None, **kwargs: Any) -> K2Observation:
        tid = task_id or TASK_IDS[(seed or 0) % len(TASK_IDS)]
        obs = self._env.reset(tid)
        if episode_id:
            self._env.episode_id = episode_id
        self._last = K2Observation(task_id=obs.task_id, prompt=obs.prompt, system=obs.system, tools=list(TOOLS), step=0, metadata={"episode_id": self._env.episode_id})
        return self._last

    def step(self, action: K2Action, timeout_s: Optional[float] = None, **kwargs: Any) -> K2Observation:
        res = self._env.step(action.to_dict())
        o = res.observation
        self._last = K2Observation(task_id=o.task_id, prompt=o.prompt, system=o.system, tools=list(TOOLS), step=o.step, last_result=o.last_result,
                                   verdict=str(res.info.get("verdict", "")), done=res.done, reward=res.reward if res.done else 0.0,
                                   metadata={k: v for k, v in res.info.items() if k != "grade"} | ({"grade": res.info["grade"]} if "grade" in res.info else {}))
        return self._last

    @property
    def state(self) -> K2State:
        s = self._env.state()
        return K2State(episode_id=s["episode_id"] or None, step_count=s["step"], task_id=s["task_id"], done=s["done"], rejections=list(s["rejections"]), grade=s["grade"])

    def close(self) -> None:
        self._env.close()


def build_app(**env_kwargs):
    return create_app(lambda: K2TasksOpenEnv(**env_kwargs), K2Action, K2Observation, env_name="k2-tasks-env")


app = build_app()
