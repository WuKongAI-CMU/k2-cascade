# External review (another agent, pasted by Peter 2026-09-25): robotics direction and gaps in the current paper

Kept verbatim below for the record. What it changes for us (decided 2026-09-25):
1. **Structured-text baseline is a real hole.** Our verbal baseline (answer + confidence word / number) is the
   weak form of a text handoff; a message listing candidate answers with probabilities is the strong form.
   Added as verbal mode `candidates` (clusters of the sender's 10 samples with their frequencies) and run on the
   377 confidence items. The theory box's bound applies only to the point-estimate message; the text must say so.
2. **Interlat (2511.09149, ACL 2026) and StateBridge (2608.13317) are hidden-state-prefix interfaces we must
   cite and, for the next paper, compare against.** Interlat already argues latents keep several hypotheses.
3. **Bytes over a network, not only GPU time:** 72 MiB per 512-token message is 6 s at 100 Mbps; the paper's
   latency claim is GPU-side only. Add the network-time sentence to the cost paragraph.
4. **Next paper's framing:** "uncertainty-aware model handoffs for embodied agents" — the receiver must choose
   act / observe more / ask / stop; two kinds of "don't know" (information vs capability); staleness; branched
   rollouts. This replaces milestone 4's "harder QA" with a partially observable discrete-skill environment.
   ALFWorld alone is not novel (Interlat); the novelty must be evidence-quality manipulation + strong text baseline.

---

(see the pasted review in the session transcript of 2026-09-25; not duplicated here to keep the repo small — key claims and arXiv ids: Fast-ThinkAct 2601.09708, Interlat 2511.09149, CloudEdgeVLA 2608.00569, Hi Robot 2502.19417, KnowNo (CoRL 2023), Where2comm (2022), StateBridge 2608.13317, Replay Gap 2608.08239, Figure Helix.)

## Second and third external reviews (same day) — corrections to make before arXiv
1. "All content-free controls sit at .46–.54" is false on the removed variant (question-only .603, random .580,
   deranged .564 vs cache .624). Report per variant, and increments over question-only with paired CIs.
2. The theory box's step from "log 3 nats" to "lower AUROC" is invalid: a binary label of "SE above median"
   would give AUROC 1 to a perfect reader. The bound is about information on the *distribution*; the measured
   gap is about how the receiver uses a label. Rewrite the box; the empirical claim stands, the deduction does not.
3. The confidence word is built from the sender's actual semantic-entropy terciles (an oracle label), not from
   the sender's own verbalisation. State it; it makes the text baseline stronger, not weaker.
4. "Drop ratio .54" is a relative drop in P(gold), not "54% of the uncertainty retained". Define it as such.
5. Heo et al.'s retention is an accuracy ratio, ours a continuation-loss gap ratio; do not conflate.
6. The deranged-sender projector answers 23.4% (chance 12.5, question-only 10.6) because its ridge base is
   content-dependent (ridge alone 21.7%); say so rather than calling it content-free.
7. Head-dimension mismatch is one of several differences for K2-0.9B; say "fails on this pair", not "requires".
8. Sender semantic entropy vs receiver first-token entropy are different measurements; the belief-structure
   metric (candidate distribution) addresses this partly; the raw→mapped→receiver probe diagnostic addresses
   where the loss occurs.
9. Bytes over a network: 72 MiB at 1 Gbps = 604 ms vs 24 ms saved; the paper must say the compute advantage
   does not translate to a network handoff at this size.
10. The current setting is context reuse (passage prefix), not a mid-task handoff of an agent's reasoning state.
