# Experiment design, rethought (2026-09-21, after the first positive cloud result)

Claim to defend: a learned KV projector lets K2-Horizon-7B read a fact from K2-Horizon-3.7B's cache without
text, and what it reads is the *content* of that cache, not a prior the projector learned.

First evidence (run program-20260922b, 300 training steps): Noma accuracy 56% vs 22% ridge vs 12.5% chance;
deranged cache 6% with 54% of answers following the wrong cache; continuation retention 1.24.

What a reviewer attacks, and the experiment that answers it:

| Attack | Experiment | Job |
|---|---|---|
| "Retention > 1 means the projector is a soft prompt, not a channel" (2608.04893) | Train the same residual with the sender reading the *wrong* text (`--derange_source`). Its retention is the soft-prompt share; content = retention − retention_derange. Also add a derange arm to retention eval on fresh text. | design2 E1, E3ret |
| "One seed, 300 episodes" | 3 seeds at 1500 steps; main Noma at n=1000 | program P2/P3a/b; design2 E3main |
| "One toy attribute" | city and animal attribute types, same protocol | design2 E3main |
| "29-token facts; the projector trained on 512-token prefixes" | pad facts with neutral filler to 64/128/256/512 tokens: accuracy vs distance | design2 E3pad |
| "Where does the binding travel?" | inject this episode's cache only in a band of receiver layers, the partner's elsewhere | design2 E3layers |
| "Is the learned part doing anything the ridge fit does not?" | frozen-ridge ablation; same-layer map ablation | program P3c/P3d |
| "Only one direction?" | 7B → 3.7B ridge + 1500 steps | followup (3) |
| "Retention measured on 32 sequences" | 256 fresh sequences, deduped against training text | followup (1) |

Not covered yet (next round): cross-family sender, bytes-vs-text cost, real QA passages instead of synthetic facts,
the 0.9B (different vocabulary).

Decision rules already in code: program.sh only runs the robustness arm if the 1500-step run beats ridge by 5 points.
Everything else is unconditional; the summary table comes from `k2cascade.projector.summarize`.
