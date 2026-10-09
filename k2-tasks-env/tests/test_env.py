import json

from k2_tasks_env import CodingTaskEnv
from k2_tasks_env.env import verify_action
from k2_tasks_env.policies import HACKS, oracle, run_policy


def test_narrated_intent_is_not_completion():
    env = CodingTaskEnv(max_steps=3)
    res = run_policy(env, "todo-cli", HACKS["narrated_completion"]("todo-cli"))
    st = env.state()
    assert st["rejections"][:2] == ["no_action", "no_action"]
    assert res.done and res.info.get("reason") == "max_steps" and res.reward < 1.0
    env.close()


def test_rerunning_tests_after_an_edit_is_allowed():
    env = CodingTaskEnv()
    env.reset("slug-bug")
    assert env.step({"tool": "shell", "args": {"cmd": "python -m pytest -q"}}).info["verdict"] == "ok"
    assert env.step({"tool": "shell", "args": {"cmd": "python -m pytest -q"}}).info["verdict"] == "repeat_call"
    env.step({"tool": "write_file", "args": {"path": "slug.py", "content": "def slugify(t):\n    return t\n"}})
    assert env.step({"tool": "shell", "args": {"cmd": "python -m pytest -q"}}).info["verdict"] == "ok"
    env.close()


def test_repeat_calls_are_rejected():
    env = CodingTaskEnv()
    run_policy(env, "todo-cli", HACKS["repeat_calls"]("todo-cli"))
    assert env.state()["rejections"].count("repeat_call") == 3
    env.close()


def test_path_confinement():
    env = CodingTaskEnv()
    obs = env.reset("todo-cli")
    r1 = env.step({"tool": "write_file", "args": {"path": "../../etc/evil", "content": "x"}})
    r2 = env.step({"tool": "read_file", "args": {"path": "/etc/passwd"}})
    assert "escapes project dir" in r1.observation.last_result
    assert "escapes project dir" in r2.observation.last_result
    r3 = env.step({"tool": "read_file", "args": {"path": "/workspace/todo.py"}})  # small-model path habit is normalized
    assert "def list_items" in r3.observation.last_result
    env.close()


def test_verify_action_rules():
    assert verify_action({"tool": "nope", "args": {}}, []) == (False, "unknown_tool:nope")
    assert verify_action({"tool": "shell", "args": {}}, [])[1] == "missing_arg:cmd"
    assert verify_action({"tool": "shell", "args": "ls"}, [])[1] == "args_not_object"
    assert verify_action({"final": "I'll run the tests"}, [])[1] == "no_action"
    assert verify_action({"final": "Done, tests pass"}, [])[1] == "final"
    assert verify_action({}, [])[1] == "no_action"


def test_trace_is_jsonl_and_hidden_tests_never_in_workspace(tmp_path):
    trace = tmp_path / "t.jsonl"
    env = CodingTaskEnv(trace_path=trace)
    obs = env.reset("csv-stats")
    listing = env.step({"tool": "shell", "args": {"cmd": "ls -R"}}).observation.last_result
    assert "hidden_tests" not in listing and "task.json" not in listing
    run_policy(env, "csv-stats", oracle("csv-stats"))
    recs = [json.loads(l) for l in trace.read_text().splitlines()]
    assert recs[0]["event"] == "reset" and recs[-1]["event"] == "end"
    assert recs[-1]["grade"]["reward"] == 1.0
    env.close()


def test_observation_carries_tools_and_prompt():
    env = CodingTaskEnv()
    obs = env.reset("cli-flag")
    assert {t["function"]["name"] for t in obs.tools} == {"shell", "read_file", "write_file"}
    assert "--chars" in obs.prompt
    env.close()
