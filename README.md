# K2 Cascade: where does the big model's advantage live?

Agents call the largest model for every step. We ran the same coding task through K2 Horizon 0.9B, 3.7B and
375B step by step and labeled which size each step actually needed. K2 Horizon's six sizes share one training recipe,
one chat template and one tool-call format, and from 3.7B up one tokenizer (250k vocab; the 0.9B has its own 64k vocab), so model size is close to the only variable, which makes
it the first open family where you can ask precisely: what does the 375B know that the 3.7B does not, and can the
small model tell before it acts?

HackCMU 2026 · Track: Optimization + IFM · built in the 24 hours of the event.

## What the traces show

All three sizes follow the same macro plan: run tests, read source, read tests, write, run tests, finish.
The difference is entirely in the arguments. The 0.9B picks the right tool every time but invents paths
(`/workspace`, `/project`, `/task`) and loops; the 3.7B takes over at the one code-writing step; the 375B is never
needed for an action. On this task the big model's advantage lives in argument and code tokens, not in tool choice.

Next: score the 375B's outputs token by token with the 3.7B (teacher forcing) to map where the surprise
concentrates (closest prior: Grotov & Malykh, arXiv 2609.05274, who do this for Qwen3 and use it as a post-hoc gate);
then train a linear probe on the 3.7B's hidden state to predict, *before* it acts, whether the step will be rejected.
If the pre-action signal matches the post-hoc judge, the small model knows its own limits and the external judge disappears.

## The idea

Coding agents call the biggest model for every step, but most steps are small: list a directory, read a file,
run the tests. K2 Cascade gives each step to the smallest K2 Horizon model first. A verifier checks the proposal
(parseable, valid tool, required arguments, not a repeat of a recent call, and a one-word yes/no from the 375B).
If it fails after one retry, the next model up takes the step and keeps control for two more steps before handing back.
Every attempt, accepted or rejected, is one JSONL line: model, step, tokens, latency, verdict, reason, thinking,
tool calls, judge verdict, and the messages the model saw. **The trace file is the dataset.**

Why K2 Horizon: six sizes, one training recipe, one chat template, one tool-call format; from 3.7B up one tokenizer
(250,624 vocab), while the 0.9B has its own 64,256 vocab. Size is close to the only variable.

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

## More tasks (added after the deadline)

Three more small tasks, each run twice in `cascade` and twice in `large` (one `large` run of csv-stats was cut off by a process restart).

| task | cascade: done / steps / escalations / who | large: done / steps |
|---|---|---|
| csv-stats (extend a summarizer) | yes / 6 / 0 / 0.9B alone · yes / 7 / 1 / 0.9B:4, 3.7B:3 | yes / 5 · yes / 7 |
| slug-bug (fix slugify) | yes / 8 / 2 / 0.9B:2, 3.7B:6 · yes / 7 / 1 / 0.9B:4, 3.7B:3 | yes / 5 · yes / 5 |
| cli-flag (argparse flags) | yes / 5 / 2 / 0.9B:1, 3.7B:1, **375B:3** · yes / 13 / 4 / 0.9B:3, 3.7B:5, **375B:5** | yes / 5 |

Three tasks, three rungs: the 0.9B alone finishes csv-stats, slug-bug needs the 3.7B, cli-flag is the first task where the 375B is needed for actions.
Rejected small-model attempts across these runs: 0.9B repeat_call ×10, judge_no ×5, unknown tool ×2; 3.7B repeat_call ×5, judge_no ×3.

Note on tokenizers: the 0.9B uses a 64,256-token vocab with BOS `<|begin_of_text|>`; 3.7B, 7B and 375B use a 250,624-token vocab with BOS `<|ifm|begin_of_text|>`. Our renderer sends the `<|ifm|begin_of_text|>` string to the 0.9B as well; it still tool-calls correctly, but the 0.9B rung is not a pure size comparison.
