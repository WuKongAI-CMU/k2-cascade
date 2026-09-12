# K2 Cascade

**Which agent steps actually need a 375B model?** A per-step cascade runner across the K2 Horizon family
(0.9B and 3.7B on a laptop via llama.cpp, 375B via the IFM API), plus the labeled step traces it produces.

HackCMU 2026 · Track: Optimization + IFM · built in the 24 hours of the event.

## The idea

Coding agents call the biggest model for every step, but most steps are small: list a directory, read a file,
run the tests. K2 Cascade gives each step to the smallest K2 Horizon model first. A verifier checks the proposal
(parseable, valid tool, required arguments, not a repeat of a recent call, and a one-word yes/no from the 375B).
If it fails after one retry, the next model up takes the step and keeps control for two more steps before handing back.
Every attempt, accepted or rejected, is one JSONL line: model, step, tokens, latency, verdict, reason, thinking,
tool calls, judge verdict, and the messages the model saw. **The trace file is the dataset.**

Why K2 Horizon: six sizes, one tokenizer, one chat template, one tool-call format, one training recipe.
Size is the only variable.

## What we measured (one task, 9 runs)

Task: a small Python repo with two failing tests; implement `--pending` / `--done` filter flags so all four pass.

| mode | done | steps | escalations | steps by model | local tok | 375B tok | judge tok | wall s |
|---|---|---|---|---|---|---|---|---|
| 0.9B only | no | 3 | – | 0.9B:3 (then loops) | 10.5k | 0 | 0 | 2 |
| 3.7B only | yes | 6 | – | 3.7B:6 | 16.2k | 0 | 0 | 23 |
| cascade ×3 | yes, yes, yes | 7 / 8 / 6 | 1 / 1 / 1 | 0.9B:4-5, 3.7B:3, 375B:0 | 25-28k | 0 | 5.6-6.8k | 33-43 |
| 375B only ×4 | yes ×4 | 5 | – | 375B:5 | 0 | 6.0-7.3k | 0 | 11-12 |

Findings on this task:

1. **The 375B was never needed for an action.** In all three cascade runs the only escalation was the
   code-writing step, and the 3.7B handled it. The 375B's whole contribution was ~6k tokens of yes/no judging.
2. **The 0.9B handles the routine steps** (list, read, run tests, write the summary) and fails in exactly one
   way: repeating its previous call (8 of 8 rejected attempts). The 3.7B's failures: one unclosed thinking block,
   two judge rejections. These are format and state errors, not reasoning errors.
3. **The cascade does not win on wall-clock here** (33-43 s vs 11-12 s all-375B): local 3.7B generation on an
   M3 Pro is the bottleneck. It wins on cloud tokens (5.6-6.8k judge-only vs 6.0-7.3k) only modestly.
   The value is the label per step, not the savings on a toy task.
4. **A harness bug we found in the 375B run**: once it replied "I'll start by running the tests" with no tool
   call, and the runner accepted that as completion. Now a step must either act or claim completion (`no_action`).

Per-attempt judge verdicts (375B on small-model proposals): 0.9B 12 yes / 0 no; 3.7B 9 yes / 2 no.

## Running

```bash
# local models: IFM's llama.cpp fork (upstream does not know the k2-horizon architecture yet)
git clone --depth 1 --branch model/K2Horizon https://github.com/MBZUAI-IFM/llama.cpp.git vendor/llama.cpp
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_METAL=ON -DLLAMA_CURL=OFF && cmake --build vendor/llama.cpp/build -j --target llama-server
vendor/llama.cpp/build/bin/llama-server -m ~/models/k2/K2-Horizon-0.9B-Q4_K_M.gguf --port 8082 -c 16384 -ngl 99 &
vendor/llama.cpp/build/bin/llama-server -m ~/models/k2/K2-Horizon-3.7B-Q4_K_M.gguf --port 8081 -c 32768 -ngl 99 &

# 375B: put IFM_API_KEY=... in ~/.config/ifm/env (or export it)
uv sync
cp -R demo/todo-cli /tmp/work && uv run python -m k2cascade.run --mode cascade --cwd /tmp/work --task "$(sed -n '3,6p' demo/TASK.md)"
uv run python -m k2cascade.compare traces/*.jsonl
```

Modes: `small` (bottom of the ladder only), `large` (375B only), `cascade`. `--no-cloud` runs a 0.9B→3.7B ladder
with no API; `--no-judge` drops the 375B yes/no check.

## What is where

- `k2cascade/prompt.py` renders prompts from IFM's own `chat_template.jinja` (with `tool_call_format=xml`, the format
  the small models actually emit regardless of instruction). Same bytes to every size.
- `k2cascade/parse.py` parses `<ifm|think>` and `<ifm|tool_calls>` output. Neither llama.cpp nor the IFM fork has a parser for it.
- `k2cascade/verify.py` rule checks; `k2cascade/cloud.py` IFM API client and the yes/no judge; `k2cascade/loop.py` the ladder.
- `k2cascade/trace.py` JSONL writer; `k2cascade/compare.py` the table above; `demo/todo-cli` the task; `traces/` the data.

## Caveats, stated plainly

- Local weights are third-party Q4_K_M quantizations (NANI-Nithin), not IFM's BF16 GGUF. Tool calling was not re-evaluated by the quantizer.
- One task, nine runs. The numbers describe this task, not K2 Horizon.
- Per-step cascading is not new (STEPWISE, R2V-Agent, TwinRouterBench, Replay Gap, all 2026). What is new here is the
  single-family ladder from 0.9B to 375B and per-attempt traces obtained by running, not replaying.
- The `/workspace/` path habit of the small models is normalized in `tools.py`; without that they loop on path errors.
