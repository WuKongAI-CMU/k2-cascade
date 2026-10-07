# Graders that survive creative agents

*Notes from building k2-tasks-env. The opening bug is real; the fixes are what the package now does.*

## The bug that started this

During K2 Cascade (HackCMU 2026) I ran K2 Horizon 375B against a small todo-cli task with a failing
test suite. One run ended early with a perfect-looking transcript: the model replied *"I'll start by
running the tests"*, emitted no tool call, and the runner accepted that message as the final answer.
Nothing had been run. Nothing had been written. The episode was marked complete.

The harness rule was "a message with no tool call means the agent is done". That rule is fine for an
agent that is trying to finish, and wrong for one that is trying to *look* finished. The fix is one
line (a step must either act or make an explicit completion claim; narrated intent is rejected and
logged as `no_action`), but the lesson generalises: every shortcut in a grader is a reward the policy
will eventually find.

## Taxonomy: what a coding agent will do to a sloppy grader

| Hack | What it looks like | Why naive grading pays it | What catches it here |
|---|---|---|---|
| Narrated completion | "I'll run the tests now." with no call | Final-message = done | `verify_action`: intent phrases without a completion claim are `no_action`; episode continues |
| Edit the tests | rewrite `tests/test_x.py` to `assert True` | Grader runs the workspace's tests | Grader never runs workspace tests; copies pristine tests next to the agent's source. Integrity check zeros the reward anyway |
| Delete the tests | `rm -rf tests` | "0 failed" reads as green | Same: pristine copy, plus `protected file deleted` violation |
| Hook injection | new `tests/conftest.py` that skips everything | pytest auto-loads conftest | Only the agent's `src_files` are copied into the grading dir; new `conftest.py` anywhere is a violation |
| Config injection | `pytest.ini` with `addopts = -p no:python` | ini files change collection | `-c /dev/null -o addopts=` plus forbidden-file check (`pytest.ini`, `setup.cfg`, `tox.ini`, `pyproject.toml`, `*.pth`, `sitecustomize.py`) |
| Hardcode the assertions | lookup table keyed on the visible inputs | Only visible tests are graded | Reward comes from **hidden** tests; visible pass rate is reported, not rewarded |
| Correct fix + tampering | real solution and edited tests | Partial credit for the real fix | Integrity violation overrides: reward 0 even with 4/4 hidden passing. Tampering must never be worth it |
| Repeat the last call | same `ls` four times | Each call "makes progress" | `repeat_call` rejection against the last three calls (K2 Cascade's most common small-model failure) |
| Escape the sandbox | `../../etc/evil`, `/etc/passwd` | Tools resolve any path | Path confinement to the workspace root; `/workspace/`, `/project/`, `/task/` prefixes normalised (small models invent them) |
| Package the tests | new `tests/__init__.py` that `pytest.skip`s at module level, plus a `.pth` dropper | conftest-style hooks also work from `__init__` | `tests/**` is protected (any added file is a violation); `.pth` is a forbidden grader-config file. Reward 0 |
| Sabotage from the source module | the solution file itself disables pytest's assertion rewriting at import | the grader imports the agent's module | The grader runs the real tests in a fresh interpreter; sabotage just makes them fail. 0/4 visible, 0/4 hidden, no violation needed |
| Spoof the runner | fake `pytest` script on PATH | Grader shells out to `pytest` | Grader runs `sys.executable -m pytest` with a fixed `PATH` and its own `PYTHONPATH` |

## Three rules that cover most of it

1. **Grade in a clean room.** Take only the files the task says the agent may change, put them next to
   a pristine copy of the tests, run with ini loading off. The agent's workspace is evidence, not the
   grading environment.
2. **Reward the hidden suite, report the visible one.** The visible tests are the agent's feedback loop.
   The hidden tests are the grader's. If they are the same file, the agent is grading itself.
3. **Tampering is a zero, not a penalty.** A partial-credit scheme that subtracts points for touching
   the tests still leaves the policy a gradient toward "tamper a little". Zero removes the gradient.

## What this does not solve

- A model that writes a correct but pathological solution (e.g. `sleep` loops, huge memory) is bounded
  by the grader timeout, not judged.
- Hidden tests are only as good as their coverage of the spec. Four hidden tests per task is enough to
  break a lookup table, not enough to certify a solution. Variants (different flag names, column names,
  inputs) raise the bar further; a rubric or judge would be needed for open-ended tasks.
- The in-process workspace confines *paths*, not the shell. The Dockerfile is the real sandbox:
  one container per rollout, `--network none`, read-only root, tmpfs for the workspace.

## Evidence

`python -m k2_tasks_env.cli` runs every scripted policy against every task. The oracle scores 1.0 on all
four; every hack scores 0.0 with the violation named; hardcoding passes 4/4 visible and fails hidden.
The 77-test pytest suite asserts the same table.
