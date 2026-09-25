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
