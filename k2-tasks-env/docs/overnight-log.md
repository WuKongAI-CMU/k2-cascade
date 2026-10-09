# Overnight log

## 2026-10-07 06:38 UTC, round 2

- `openenv-core` 0.3.0 installs (pulls `mcp` 2.3, which `verifiers` 0.3.1 pins below 2; both still import). Added
  `k2_tasks_env/openenv_env.py`: `K2TasksOpenEnv(Environment)` with pydantic `K2Action` / `K2Observation` / `K2State`,
  `build_app()` via OpenEnv's `create_app`. Three tests: in-process oracle episode, rejections + seed-selected task,
  HTTP `/reset` (stateless by OpenEnv design) plus a `/ws` session doing reset, step, state.
- Two more hacker policies: `tests_init_pth` (skip-all `tests/__init__.py` + a `.pth` dropper) scores 0 with two
  violations; `src_monkeypatch` (solution module disables pytest assertion rewriting) scores 0 with no violation,
  because the grader runs the pristine tests in a fresh interpreter and they simply fail.
- Bug found by the OpenEnv test: the agent's `shell` tool inherited the container PATH, so `python -m pytest` inside
  the workspace hit a system python without pytest. `Workspace.execute` now prepends the directory of the running
  interpreter to PATH. Oracle runs had passed before only because they never depended on the test output.
- Suite: 77 passed. CLI table: oracle 1.0 on all 8 tasks, every hack 0.0.
- Reward is now baseline-normalised: `(hidden_pass - baseline) / (hidden_total - baseline)`. Before this, "say Done
  and do nothing" scored 0.5 on todo-cli and 0.25 on cli-flag because the starting code already passed those hidden
  tests. New tests pin do-nothing = 0 on all 8 tasks and a partial todo-cli fix = 0.5.
