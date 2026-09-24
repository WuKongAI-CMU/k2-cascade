# GOAL — the living ladder (north star / 12-month / quarter / week). Reviewed weekly; history in goal/LOG.md

The ladder has four rungs. Each rung is judged only by the rung below it; the top rung is never "done".

## North star (decade): the agent economy runs on state, not text
Populations of models that trade work with each other and improve from it: an agent knows what it cannot do,
hands its *state* (not a transcript) to a stronger or more specialised one, pays for it, and learns from what
comes back. Three things such an economy needs that nobody has: a channel for state between different models
(what we built), a way to carry how sure the sender was so the buyer is not defrauded by confident-looking
handoffs (what we are measuring; "confidence laundering" is the fraud), and a price — bytes, latency, accuracy
per dollar — so the trade clears. We do not claim these are sufficient for self-improving systems; we claim they
are necessary and testable at our scale, and that the papers come out of the same experiments.

## 12-month line (to 2027-09): state channels that carry uncertainty
Already set in docs/goal-2026-09-21.md; status 2026-09-24: milestone 1 done (channel exists, 4 pairs, controls),
milestone 2 done (SQuAD, HotpotQA different-context, compression), milestone 3 half done (uncertainty survives
attenuated; cache beats a verbal confidence word; two training fixes tried, second running). Milestone 4
(handoff policy, accuracy per dollar) not started.

## Quarter (to 2026-12-31)
1. arXiv v1 by 2026-10-01; ICML 2027 submission 2027-01-28.
2. Milestone 3 closed: a projector that keeps ≥ 80% of the sender's confidence drop, or a proof that the
   receiver-side objective cannot (then the fix moves to the sender).
3. Milestone 4 started: probe + channel cascade on one real task, three baselines, accuracy per dollar.
4. One thing that feeds the north star directly: state as a tradable good. Price a handoff (bytes × latency ×
   accuracy gained) and show a two-agent market where buying the small agent's state beats buying its text at
   the same price. Pilot only; the design doc precedes any run.

## Week (rolling; rewritten every Monday)
Week of 2026-09-22: all twelve experiment blocks run; paper v2 compiled (abstract tightened); overnight:
entropy-matched projector, numeric verbal baseline, seeds for Qwen and HotpotQA; milestone-4 code written
(cascade.py, policy.py) and its first run launched (600 SQuAD items, probe vs thresholds, state vs text vs
verbal handoff, cost in measured latency). 09-24 morning: all overnight results in (two receiver-side fixes for
the uncertainty loss failed; numeric verbal baseline still loses to the cache; Qwen 3 seeds 71.7±4.5; HotpotQA
3 seeds 39.9±2.1); milestone 4 killed on SQuAD (receiver only 3.5 F1 better); paper v3 being compiled.
Week of 2026-09-29: read-through and rewrite of abstract/intro/limitations; arXiv v1; start milestone 4 code;
one-page design note for the priced-handoff pilot (ties to docs/agent-communication-futures-2026-09-21.md).

## Review protocol
- Weekly (Monday): rewrite the Week rung; move anything finished up into Quarter status; note what was killed.
- Quarterly: rewrite Quarter and re-read the 12-month line against results; the 12-month line may be replaced,
  the north star only re-worded.
- Every result gets its control before its headline; every rung has a kill criterion or it is not a goal.
- Spending: standing GCP allowance $500 total (Peter, 2026-09-21); above it, quote first.
- Anything outward-facing (arXiv, emails, posts) is Peter's button.
