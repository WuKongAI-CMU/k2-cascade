# K2 Cascade — 3-minute pitch (matches demo/slides-v2.html, 8 slides)

Before you start: both local servers up (`curl localhost:8081/health`, `localhost:8082/health`), a fresh copy of demo/todo-cli ready, terminal font large.

---

## 1 · The end state (0:00–0:25)

Today every agent step goes to the biggest model. Not because it needs to, but because no model can tell when it's out of its depth.
We think the next capability isn't a bigger model. It's a model that knows its own limits, and only then escalates, delegates, or asks.
K2 Cascade is the first measurement toward that: which steps need a big model, and could the small one have known?

## 2 · The question (0:25–0:40)

Concretely: a coding agent. Most steps are list a directory, read a file, run the tests. We built a ladder: 0.9B and 3.7B run on this MacBook, 375B through the IFM API.

## 3 · Why K2 Horizon (0:40–0:55)

IFM open-sourced K2 Horizon last week. Six sizes, one training recipe, one chat template, one tool-call format, and from 3.7B up one tokenizer. Size is very close to the only variable. That's what makes the question measurable.
Each step goes to the smallest model first. A verifier checks it. On failure the next size takes over. Every attempt is one line in a file. The trace file is the dataset.

## → switch to terminal (0:55–1:35)

```
cd ~/projects/k2-cascade && cp -R demo/todo-cli /tmp/demo-$(date +%s) && uv run python -m k2cascade.run --mode cascade --cwd $(ls -d /tmp/demo-* | tail -1) --task "$(sed -n '3,6p' demo/TASK.md)"
```

While it runs: "Two failing tests, implement two flags. Watch the 0.9B: runs tests, reads the source, reads the tests. Step four, writing the code: rejected, the 3.7B takes it. The 375B only ever answers yes or no." (~40 s; if the network dies, add `--no-cloud`.)

## 4 · Nine runs, one task (1:35–2:00)

Nine runs. Three cascade runs, all finished, each with exactly one escalation, at the code-writing step, to the 3.7B.
Steps where the 375B was needed for an action: zero. Its entire contribution was six thousand tokens of yes/no judging.
Honest number: the cascade is slower, 33 to 43 seconds against 11 for the 375B alone. Local generation is the bottleneck. The value is the label on every step, not the savings on a toy task.

## 5 · What the traces say (2:00–2:30)

All three sizes follow the same plan. The 0.9B picks the right tool every time, then invents paths and repeats itself. The 375B once narrated an action without calling a tool; the runner now rejects that.
Then we scored the 375B's steps token by token with the 3.7B: 0.01 nats per token on tool arguments, same as its own output. The surprise is in the prose, not the actions. The big model talks better; it doesn't act differently.
After the deadline we added three tasks: csv-stats, the 0.9B alone; slug-bug, needs the 3.7B; cli-flag, the first that needs the 375B. Three tasks, three rungs.

## 6 · Inside the model (2:30–2:45)

This is what we build next. Same tokens into the 3.7B, take the hidden state at the last position, a linear probe outputs the probability this step gets rejected, a gate decides: act, escalate, or ask. Labels already exist in today's traces. Only the probe trains, never the model. Below it, later: the 0.9B's KV cache projected straight into the 3.7B so it continues instead of re-reading.

## 7 · Roadmap (2:45–2:55)

Six rungs, one number each. Two done today, one running in the background, the pre-action probe next week.

## 8 · Close (2:55–3:00)

When the internal signal matches the external judge, the judge disappears. That's an agent that knows its limits. Repo is public: github.com/WuKongAI-CMU/k2-cascade.

---

## If they ask

- **"Isn't this just model routing?"** Routing decides after the fact with an outside judge. We want the decision before the act, from inside the model. A September paper (arXiv 2609.05274) does the after-the-fact version; the before version is open.
- **"Why did the 375B never get used?"** Because the task had one hard step and the 3.7B could do it. That's the finding, not a failure. On cli-flag it was needed for 3 to 5 steps.
- **"One tokenizer?"** From 3.7B up. The 0.9B has its own 64k vocab, so that rung isn't a pure size comparison. It's in the README.
- **"What was hard?"** Upstream llama.cpp doesn't know the architecture, we built IFM's fork. No parser exists for K2's tool-call format, we wrote one, and found the small models emit XML no matter what you ask for.
- **"Cascade is slower, so what did you optimize?"** Which steps get which size. The token savings on a toy task are small; the per-step label is what we were after.
