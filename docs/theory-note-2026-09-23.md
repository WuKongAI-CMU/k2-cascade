# Theory box: why a text handoff launders uncertainty, and what a state handoff can keep (draft, 2026-09-23)

**Setting.** Sender S reads context c and holds a belief over answers, a distribution p_S(y | c). It hands off to
receiver R through a channel m = f(c, S). R then produces p_R(y | m, q). We ask how much of p_S survives in m.

**Text handoff is a point estimate.** With m = argmax_y p_S(y | c) (or one sample), m is a function of p_S that
takes at most |Y| values. The information m can carry about p_S is bounded by log|Y| nats regardless of how
peaked or flat p_S is; two senders with the same mode and entropies 0.1 and 2.0 send the same message. Adding a
verbalised confidence word adds at most log k nats (k levels; k = 3 here) and relies on the sender's verbal
calibration, which is poor for base models (Tian et al. 2023; Yona et al. 2024). This is the mechanism behind
"confidence laundering" (2606.20662): the receiver cannot be more informed about the sender's uncertainty than
the message allows, and it treats a confident-looking point estimate as a fact.

**A state handoff carries the computation, not its summary.** With m = KV cache of S at every layer and position,
m is a sufficient statistic for S's next-token distribution (S's own decoder recovers p_S(y | c) exactly from it),
so I(m; p_S) is not bounded by log|Y|. What the *receiver* recovers depends on the map g: cache_S → cache_R and
on R's ability to read a foreign cache. Two losses stack:
- *map loss*: g is fit on continuation likelihood; nothing forces g to preserve the spread of p_S. A learned map
  trained on point targets contracts predictive variance (the distillation result of 2601.18909). Our measured
  54% confidence-drop ratio is this term.
- *read loss*: R reads the mapped cache with its own heads; the binding lives in R's later layers (layer-band
  table), so whatever g places elsewhere is ignored.

**Prediction that the experiments test.**
1. Verbal handoff cannot exceed the log k bound: its AUROC against the sender's semantic entropy should sit near
   the tercile-label ceiling, and should not improve with more items.
2. The cache handoff's AUROC can exceed the verbal one and approach the text-reading receiver's, because it is
   not information-limited; its shortfall is map loss, so a map trained to match the receiver's own text-path
   distribution (KL term) should raise the drop ratio toward 1 without changing accuracy.
3. Neither the receiver reading text nor reading the cache can exceed the sender-side ceiling set by how well
   R's first-token entropy correlates with S's semantic entropy at all (the two models disagree about which
   questions are hard); this is why the absolute AUROC is ~0.6 and why the comparison must be relative.

**What would falsify the framing.** If the KL-trained map raises the drop ratio but the verbal arm matches the
cache arm on AUROC, the extra capacity of the state channel is not being used for uncertainty, and the honest
claim shrinks to "state handoff preserves content; uncertainty transfer is no better than a confidence word".
