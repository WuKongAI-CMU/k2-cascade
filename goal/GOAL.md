# GOAL — the living ladder (north star / 12-month / quarter / week). Reviewed weekly; history in goal/LOG.md

The ladder has four rungs. Each rung is judged only by the rung below it; the top rung is never "done".

## North star (decade): the agent economy runs on state, not text

Four sentences (2026-09-25), from most to least certain:
1. The unit of computation is moving from one model to a population of models (routing, cascades, agents,
   speculative decoding). Their protocol is still text — human-readable by inertia, not by design.
2. The next protocol is state. Three things text cannot carry: the computation already done, the shape of a
   belief (which alternatives), and intermediate representations that are not words. Inside robots this already
   exists (Helix, GR00T); across independent models it does not. We are building the earliest version.
3. Once state can be handed off there is an economy: state has cost (bytes, latency), value (reuse, accuracy)
   and fraud (confident wrong state). Cost, value, fraud → price, reputation, audit. Our three pieces are the
   three primitives: the channel, uncertainty that travels with it, accuracy per dollar.
4. (bet) Self-improving systems will not be one model getting stronger but a population handing state to each
   other and learning from what comes back. Knowing you cannot (the probe) and handing off state (the channel)
   are the first two steps of that loop.

Populations of models that trade work with each other and improve from it: an agent knows what it cannot do,
hands its *state* (not a transcript) to a stronger or more specialised one, pays for it, and learns from what
comes back. Three things such an economy needs that nobody has: a channel for state between different models
(what we built), a way to carry how sure the sender was so the buyer is not defrauded by confident-looking
handoffs (what we are measuring; "confidence laundering" is the fraud), and a price — bytes, latency, accuracy
per dollar — so the trade clears. We do not claim these are sufficient for self-improving systems; we claim they
are necessary and testable at our scale, and that the papers come out of the same experiments.

## 12-month line (to 2027-09): state channels that carry uncertainty
Already set in docs/goal-2026-09-21.md; status 2026-09-24: milestone 1 done (channel exists, 4 pairs, controls),
milestone 2 done (SQuAD, HotpotQA different-context, compression), milestone 3 run and its pre-registered proceed
criterion missed (docs/design-confidence-2026-09-22.md asked for cache-arm AUROC ≥ .70; measured .585 / .605 / .624;
the drop-ratio condition passed at .54), so the design document's fallback reading holds: content transfers,
uncertainty does not; cache > verbal word is SUGGESTED, not SHOWN; two training fixes tried, neither moved it
(thesis ch. "Uncertainty", paragraph "The pre-registered criterion was missed", 2026-10-07). Milestone 4
(handoff policy, accuracy per dollar) not started.

## Quarter (to 2026-12-31)
Rewritten 2026-10-07 when Part IV of docs/thesis/thesis.tex was adopted as the program. The quarter is the
first two program chapters, each with the kill written there, plus the two outward dates.
1. Dates: arXiv v1 (was due 2026-10-01; still the first outward item, Peter's button); ICML 2027 submission
   2027-01-28 (goal/LOG.md 09-25 named COLM 2027 first choice; the thesis records the disagreement; this
   file keeps ICML until Peter picks).
2. Program chapter 1 — the pre-action probe (thesis ch. "Knowing you cannot"). Done: AUROC .749 on 600 SQuAD
   items vs semantic entropy .689; .876 [.813, .944] on 127 agent steps vs metadata .704. This quarter:
   train on SQuAD clean, test on HotpotQA partitioned and the agent steps without refitting; split failures
   into information-missing (removed variant) vs capability-missing (clean but wrong), per-split AUROC with
   passage-grouped folds. Eval-only on cached states. Kill: off-distribution AUROC ≤ .69 on the same items,
   or the two kinds not separable (per-split AUROCs inside each other's interval) → the trigger is a
   within-dataset shortcut and later chapters use semantic entropy as the trigger.
3. Program chapter 2 — the channel and a priced catalogue (thesis ch. "The channel"). Done: binding 63.0%
   (chance 12.5, deranged 4.6, deranged-sender projector 23.4), SQuAD 53.0 vs 54.7 text, HotpotQA 39.9 ± 2.1
   vs 45.8, Qwen3-4B 71.7 ± 4.5, 0.9B 17.0 (fails). This quarter: the week-1 no-GPU diagnostic (attention-
   output similarity and KV R² on the 20 existing compression rows, rank-correlated with SQuAD F1); the
   catalogue's four columns defined (bytes/token, latency vs re-prefill, binding with derange, audit AUROC);
   the learned-codec block priced before it runs (slots {16, 64, 256}, packed int4, wire bytes). Kill:
   geometry/vocabulary features do not predict transfer above a shuffled baseline → "transfer is a property
   of the triple" is dropped; nothing ≤ 1 MB per message keeps SQuAD F1 within 5 of text with the derange arm
   near chance → the channel stays a measurement instrument.
Order of runs is the thesis's "order of the kills": instruction-tuned receiver on the word arm and the three
projector seeds through the confidence protocol come first (block 1), because only they can change the
arXiv abstract. Standing allowance $500 total; codec training is quoted to Peter before it starts.

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
