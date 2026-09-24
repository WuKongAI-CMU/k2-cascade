# Milestone 4 design: the handoff policy, priced (2026-09-24, not yet run)

**Question.** A small model answers; when it predicts it will fail, it hands off. Does handing off its *state*
beat handing off its *text*, and beat always using the big model, per dollar?

**Task.** SQuAD v1.1 validation, 600 items (the shared file). Sender K2-3.7B, receiver K2-7B, projector seed 2.

**Sender side (one pass, `cascade.py --stage sender`).** For each item: the 3.7B reads passage + question,
answers greedily (≤ 8 tokens), and we record (a) F1 of its answer, (b) its first-token max probability and
entropy, (c) its semantic entropy from 10 samples (already computed for the confidence items; recompute here
for all 600), (d) the hidden state at the last prompt token at layers 12 / 18 / 24 (the pre-action probe
position from the agent-trace work). Also its self-read latency.

**Receiver side (reuse `qa.py --per_item`).** For each item: the 7B's answer and F1 (i) reading the passage
(text handoff = re-prefill), (ii) reading the mapped cache (state handoff), (iii) reading the sender's answer
plus confidence word (verbal handoff). All three already run for the 377 confidence items; run for the 600.

**Policy (`policy.py`, no GPU).** A logistic probe on (d) predicting "sender F1 < 0.5", trained with nested
leave-one-fold-out over 5 folds (reuse `nested.py`'s protocol); baselines for the trigger: max-prob threshold,
semantic-entropy threshold, random at the same handoff rate. For each trigger and each handoff rate r in
{0, 0.1, ..., 1}: accuracy = mean F1 of the chosen answer; cost = (1 − r)·c_small + r·(c_small + c_handoff),
with c from measured latencies (sender read 45 ms, receiver re-prefill 48 ms, projector 24 ms; the receiver's
answer decode is the same in every arm and is omitted) or from a price sheet (per-token API prices, appendix).
Curves: accuracy vs cost for state / text / verbal handoff, with always-small and always-big as endpoints and
the oracle trigger (hand off exactly the items the sender gets wrong) as the ceiling.

**Claims the curves can support.** (1) At equal cost, state handoff ≥ text handoff (the projector is 2× faster
than re-prefill, so at any handoff rate the state curve sits left of the text curve; the question is whether it
also sits above, given F1 53.0 vs 54.7). (2) The probe trigger beats confidence-threshold triggers (the earlier
nested result: +0.14 AUC over metadata for the 3.7B). (3) A verbal handoff is dominated.

**Pricing pilot (north-star item 4).** Treat the sender's cache as a good: price p_state = bytes × $/byte +
latency × $/s; price p_text likewise. A buyer (the receiver) with a budget chooses per item which to buy; report
buyer's F1 per dollar under both goods across budgets. This is the same curve with cost in dollars; the "market"
is the budget sweep. Design only; run after milestone 4's curves exist.

**Kill.** If state handoff never beats text handoff at equal cost on any rate, milestone 4's claim is only
"the probe trigger helps", and the pricing pilot is dropped.

**Cost.** Sender pass 30 min, receiver passes 3 × 40 min, one A100, under $10. Code: `cascade.py` (sender
features), `policy.py` (curves, CIs), one job script.
