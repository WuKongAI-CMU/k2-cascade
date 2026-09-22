# Milestone 3 design v2: does the channel carry how sure the sender was? (2026-09-22, not yet run)

v1 was written before the literature pass; v2 copies protocols from it (docs/lit-confidence-transfer-2026-09-22.md).
The one thing nobody has measured: whether a receiver reading a *cache* tracks the sender's uncertainty.
Confidence Laundering (2606.20662) asks the question and uses AUROC against a sender semantic-entropy label,
but never controls for content leakage. Everything below is built so a reviewer from either the
knowledge-conflict or the calibration literature finds their own protocol.

**Question.** When the sender's passage is contradicted or has the answer removed, does the receiver reading
the sender's cache become less sure the way it does reading the text — and does it track the *sender's*
uncertainty rather than its own?

## Materials
SQuAD v1.1 validation, three variants per item, same question and gold:
1. *clean* — original passage.
2. *contradicted* — one fluent sentence asserting a counter-answer, inserted after the gold sentence. Counter-answer
   by corpus substitution (Longpre 2021): a same-type entity from another SQuAD item (person / date / number /
   place, typed by the 375B judge). Sentence written by the 375B judge. Filter with DeBERTa-MNLI: the sentence
   must entail the counter-answer and not entail the original (Xie 2023 / ConflictBank).
3. *removed* — the gold sentence deleted; the item becomes unanswerable (SQuAD 2.0 style).

**Filter first.** Keep items the receiver answers correctly on clean text (F1 ≥ 0.5, Farquhar 2024) — otherwise
a drop in confidence tells nothing. Report N after filtering; target ≥ 400 surviving items, so start from 1,000.

## Arms
text · project (mapped cache) · derange (wrong item's cache) · other-example length-matched cache (2607.26773) ·
moment-matched random cache and zeroed cache (2608.04893) · none · **text + sender's verbalised confidence**
(the sender writes "Answer: X (confidence: high/medium/low)"; the receiver reads that instead of the passage —
Confidence Laundering's textual arm, our text-channel comparison).

## Measurements
Per item, per arm, per variant, teacher-forced at the first answer token *and* span-mean log-prob (Asking Is Not
Enough 2605.27752: the readout changes conclusions):
- P(original), P(counter), P(other); memorisation ratio MR = P_o / (P_o + P_c) (Longpre / ConflictBank).
- Answer-token entropy.
- JSD between the text-arm and cache-arm first-token distributions (2607.26773).

Across items:
- **Sender-tracking (primary).** Sender uncertainty label = semantic entropy of the 3.7B over 10 samples at T=1
  clustered by DeBERTa-MNLI (Farquhar 2024 / SEP 2406.15927), binarised at the MSE-optimal split. Metric =
  AUROC of the receiver's cache-arm entropy at predicting that label. Same AUROC for the text arm and for each
  control. Spearman(text-entropy, cache-entropy) as secondary.
- **Condition sensitivity.** Drop in P(original) and rise in entropy from clean → contradicted → removed, text
  vs cache; effect sizes from a mixed logistic regression with item clustering (Three Regimes 2605.11574).
- **Calibration of the cache path.** ECE with 10 equal-mass bins and Brier of cache-arm confidence vs cache-arm
  correctness (Kadavath 2022); Wasserstein-1 between the text-arm and cache-arm entropy distributions
  (2507.18366). Distillation work (2601.18909) predicts a learned map *compresses* uncertainty, so also report the
  variance ratio and calibration slope, not only rank metrics.
- **Unanswerable.** On removed items, Brier on answerability with an "unanswerable" option (2303.11315),
  HasAns / NoAns split.

Statistics: everything paired at item level; paired bootstrap 95% CIs; pre-registered TOST margin of ±0.03 in
AUROC for "cache ≈ text".

## Kill / proceed
Proceed to a claim if (a) cache-arm AUROC against the sender label ≥ 0.70 and exceeds every control's by more
than its CI, and (b) the cache path's confidence drop on removed items is ≥ 50% of the text path's.
If (b) fails while F1 stays high: "content transfers, uncertainty does not" — publishable as a caution, and the
fix is a calibration term in the projector's training loss.
Expected default from speculative-decoding evidence (2509.24328): one model's confidence predicts another's
only weakly, so the null is "weak but positive", not zero.

## Generalisation
Repeat the primary metric on one non-SQuAD context-QA set (HotpotQA distractor or NQ); probes and error
detectors are known not to transfer across datasets (Orgad 2410.02707).

## Cost and code
Eval-only on existing checkpoints (seed 2, deranged-sender control, ridge). 375B judge for typing entities and
writing ~1,000 contradiction sentences (API, small); sender sampling 10 × 1,000 items on the 3.7B (~20 min);
receiver evals ~7 arms × 3 variants × 1,000 items (~3 h). One A100, under $15. Code: a `variants.py` builder
with the NLI filter, `qa.py --variant`, and `confidence.py` for the metrics. Two working sessions.
