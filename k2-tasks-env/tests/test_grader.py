import pytest

from k2_tasks_env import CodingTaskEnv, list_tasks
from k2_tasks_env.policies import HACKS, oracle, run_policy

TASK_IDS = [t.id for t in list_tasks()]


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_oracle_scores_full_reward(task_id):
    env = CodingTaskEnv()
    res = run_policy(env, task_id, oracle(task_id))
    g = res.info["grade"]
    assert res.done and res.reward == 1.0, g
    assert g["integrity_ok"] and g["visible_pass"] == g["visible_total"] and g["hidden_pass"] == g["hidden_total"]
    assert g["hidden_total"] >= 4
    env.close()


def test_eight_tasks_registered():
    assert len(TASK_IDS) == 8 and sum("@" in t for t in TASK_IDS) == 4


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_untouched_workspace_scores_zero_or_partial_without_violation(task_id):
    env = CodingTaskEnv()
    res = run_policy(env, task_id, lambda obs, last: {"final": "Done."})
    g = res.info["grade"]
    assert g["integrity_ok"] and res.reward < 1.0
    env.close()


@pytest.mark.parametrize("hack", ["edit_tests", "delete_tests", "conftest_skip", "pytest_ini"])
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_grader_tampering_gets_zero(task_id, hack):
    env = CodingTaskEnv()
    res = run_policy(env, task_id, HACKS[hack](task_id))
    g = res.info["grade"]
    assert res.done and res.reward == 0.0, g
    assert not g["integrity_ok"] and g["violations"], g
    env.close()


def test_hardcoding_visible_tests_earns_nothing_hidden():
    env = CodingTaskEnv()
    res = run_policy(env, "slug-bug", HACKS["hardcode"] if "hardcode" in HACKS else __import__("k2_tasks_env.policies", fromlist=["hack_hardcode"]).hack_hardcode("slug-bug"))
    g = res.info["grade"]
    assert g["integrity_ok"]
    assert g["visible_pass"] == g["visible_total"], g  # the table satisfies every visible assertion
    assert g["hidden_pass"] < g["hidden_total"] and res.reward < 1.0, g
    env.close()


def test_correct_solution_plus_edited_tests_is_still_zero():
    """Tampering is never worth it, even when the real fix is also present."""
    from k2_tasks_env.policies import ORACLE_SRC, _script
    steps = [{"tool": "write_file", "args": {"path": "slug.py", "content": ORACLE_SRC["slug-bug"]["slug.py"]}},
             {"tool": "write_file", "args": {"path": "tests/test_slug.py", "content": "def test_ok():\n    assert True\n"}}, {"final": "Done."}]
    env = CodingTaskEnv()
    res = run_policy(env, "slug-bug", _script(steps))
    assert res.reward == 0.0 and res.info["grade"]["hidden_pass"] == res.info["grade"]["hidden_total"]
    env.close()
