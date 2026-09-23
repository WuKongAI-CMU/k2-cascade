# Gap checklist for the arXiv / ICML paper (2026-09-22)

Against docs/lit-venue-standards-2026-09-22.md. "Have" = already in analysis/cloud; "Run" = needs a job; "Write" = only text.

| Bar (paper that sets it) | Status | What to do |
|---|---|---|
| Receiver-alone and text baselines on the same items (C2C) | Have | none / text rows in every table |
| Boundary-replacement controls: deranged, content-free, wrong-example (2607.26773) | Have | deranged cache, deranged-sender projector, zero, moment-matched random (qa.py) |
| Decompose aggregate into "content" vs "message present" (2607.26773) | Write | report content-free projector as the "message present, wrong content" term |
| Cost in bytes and adapter parameters (LCF 13 MB) | Write + Run | projector 151M params (~600 MB bf16) — large; report it; compression run: layers 18–35 only, top-k heads, low-rank |
| Latency vs re-prefill and vs text (closed-form 2.7–25x) | Run | time the 7B prefill of 512 tokens vs projector forward; cheap, eval-only |
| ≥3 pairs incl. cross-family, failures reported (closed-form: 6 pairs, 2 collapse) | Run | reverse 7B→3.7B done; add 0.9B→7B (vocab differs: needs re-tokenised sender text) and one non-K2 sender (Qwen3-4B → K2-7B) |
| Oracle / upper bound (C2C, closed-form) | Have | self arm = text; oracle in retention |
| Different-context setting (LCF-X, partitioned HotpotQA) | Run | sender reads passage A, receiver reads passage B + question; two-hop |
| Which layers carry the signal (closed-form, LCF) | Have | layer-band table |
| Linear vs nonlinear mapper (closed-form) | Have | ridge 22% vs trained 73%; frozen-ridge 59% |
| Multiple benchmarks (2607.26773) | Run | HotpotQA partitioned; possibly ARC-C style MC with sender-only context |
| Seeds and CIs | Have (partly) | 3 seeds on binding; add paired-bootstrap CIs everywhere (confidence.py has the helper) |
| Confidence transfer (nobody) | Run | confidence.sh, waiting on variants |

Order of runs: confidence → compression + latency (same VM, eval-only) → HotpotQA partitioned → cross-family.
