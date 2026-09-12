# K2 Cascade

Which agent steps need a 375B model, and which does a 3.7B handle? A per-step cascade runner across the
K2 Horizon family (0.9B / 3.7B locally via llama.cpp, 375B via the IFM API), plus the labeled step traces it produces.

HackCMU 2026, track: Optimization + IFM.

## What it does

Each agent step (one tool call) is first proposed by the small model running on the laptop. A cheap verifier
checks the proposal (parseable, valid tool, required arguments, not a repeat). If it fails after one retry, the
375B takes that single step and hands control back. Every attempt, accepted or not, is written as one JSONL line:
model, step, tokens, latency, verdict, reason, thinking, tool calls, and the messages the model saw.

Three modes on the same task: `small` (3.7B only), `large` (375B only), `cascade`.

## Why K2 Horizon

Same tokenizer, chat template, tool-call format and training recipe across six sizes, so size is the only variable.

## Running

```bash
# local 3.7B (IFM llama.cpp fork, upstream does not know the architecture yet)
vendor/llama.cpp/build/bin/llama-server -m ~/models/k2/K2-Horizon-3.7B-Q4_K_M.gguf --port 8081 -c 32768 -ngl 99

export IFM_API_KEY=...   # 375B
uv run python -m k2cascade.run --task "..." --cwd path/to/repo --mode cascade
uv run python -m k2cascade.compare traces/*.jsonl
```

## Caveats

- Local weights are third-party Q4_K_M quantizations (NANI-Nithin), not IFM's BF16 GGUF.
- Prompts are rendered from IFM's own `chat_template.jinja` with `tool_call_format=json`; the model's
  `<ifm|tool_calls>` output is parsed by `k2cascade/parse.py`, since neither llama.cpp nor its IFM fork has a parser for it.
- Per-step cascading itself is not new (STEPWISE, R2V-Agent, TwinRouterBench, Replay Gap, 2026). What is new here is
  the single-family ladder and the released per-attempt traces.
