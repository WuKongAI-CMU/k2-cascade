# Goal ladder (set 2026-09-24; reviewed weekly, rewritten quarterly)

The ladder has four rungs. Each rung is judged only by the rung below it; the top rung is never "done".

## North star (decade)
Systems of models that improve themselves: they know what they cannot do, hand their *state* to something
stronger, and learn from what comes back. Communication of state and calibrated self-knowledge are the two
ingredients we are betting on; we do not claim they are sufficient for recursive self-improvement, only
that nobody has them yet and both are testable at our scale.

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
4. One thing that feeds the north star directly: a receiver that is *worse* at a task learns from the sender's
   cache without gradient steps (in-context transfer of skill, not fact). Pilot only.

## Week (rolling; rewritten every Monday)
Week of 2026-09-22: all twelve experiment blocks run; paper v2 compiled; overnight: entropy-matched projector,
numeric verbal baseline, seeds for Qwen and HotpotQA.
Week of 2026-09-29: read-through and rewrite of abstract/intro/limitations; arXiv v1; start milestone 4 code.

## Review protocol
- Weekly (Monday): rewrite the Week rung; move anything finished up into Quarter status; note what was killed.
- Quarterly: rewrite Quarter and re-read the 12-month line against results; the 12-month line may be replaced,
  the north star only re-worded.
- Every result gets its control before its headline; every rung has a kill criterion or it is not a goal.
- Spending: standing GCP allowance $500 total (Peter, 2026-09-21); above it, quote first.
- Anything outward-facing (arXiv, emails, posts) is Peter's button.
