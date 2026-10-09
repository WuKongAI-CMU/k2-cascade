import pytest

openenv = pytest.importorskip("openenv")

from k2_tasks_env.openenv_env import K2Action, K2TasksOpenEnv, build_app
from k2_tasks_env.policies import ORACLE_SRC


def test_openenv_oracle_episode():
    env = K2TasksOpenEnv()
    obs = env.reset(task_id="slug-bug")
    assert obs.task_id == "slug-bug" and {t["function"]["name"] for t in obs.tools} == {"shell", "read_file", "write_file"}
    o = env.step(K2Action(tool="shell", args={"cmd": "python -m pytest -q"}))
    assert o.verdict == "ok" and not o.done and "failed" in o.last_result
    o = env.step(K2Action(tool="write_file", args={"path": "slug.py", "content": ORACLE_SRC["slug-bug"]["slug.py"]}))
    o = env.step(K2Action(final="Fixed slugify; all tests pass."))
    assert o.done and o.reward == 1.0 and o.metadata["grade"]["hidden_pass"] == 4
    assert env.state.step_count == 3 and env.state.done and env.state.grade["integrity_ok"]
    env.close()


def test_openenv_rejections_and_seed_selects_task():
    env = K2TasksOpenEnv(max_steps=2)
    obs = env.reset(seed=1)
    assert obs.task_id == "cli-flag@longest" or obs.task_id in [t for t in __import__("k2_tasks_env.openenv_env", fromlist=["TASK_IDS"]).TASK_IDS]
    o = env.step(K2Action(final="I'll start by running the tests."))
    assert o.verdict == "no_action" and not o.done
    o = env.step(K2Action(tool="read_file", args={"path": "/etc/passwd"}))
    assert "escapes project dir" in o.last_result and o.done  # max_steps reached -> graded
    assert env.state.rejections == ["no_action"]
    env.close()


def test_openenv_http_reset_is_stateless_and_ws_session_is_stateful():
    """OpenEnv's plain HTTP /reset and /step each use a throwaway env; episodes live on the /ws session."""
    from fastapi.testclient import TestClient

    client = TestClient(build_app())
    r = client.post("/reset", json={"task_id": "csv-stats"})
    assert r.status_code == 200, r.text
    assert r.json()["observation"]["task_id"] == "csv-stats"

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {"task_id": "csv-stats"}})
        first = ws.receive_json()
        assert first["type"] == "observation" and first["data"]["observation"]["task_id"] == "csv-stats"
        ws.send_json({"type": "step", "data": {"tool": "shell", "args": {"cmd": "ls"}}})
        step = ws.receive_json()
        obs = step["data"]["observation"]
        assert "stats.py" in obs["last_result"] and "hidden_tests" not in obs["last_result"]
        ws.send_json({"type": "state"})
        st = ws.receive_json()
        assert st["type"] == "state" and st["data"]["task_id"] == "csv-stats" and st["data"]["step_count"] == 1
        ws.send_json({"type": "close"})
