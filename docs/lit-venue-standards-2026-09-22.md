*Research-agent pass, 2026-09-22: how the comparison papers are built and what ICML requires. OpenReview was bot-blocked, so reviewer text is missing.*

Budget is spent (15 calls); the OpenReview API redirected to the same challenge page, so reviewer text for C2C is unavailable in this pass. Report follows.

## (a) The four papers

**Cache-to-Cache (C2C), arXiv 2510.03215** — https://arxiv.org/abs/2510.03215
- Venue: ICLR 2026 (arXiv comments field; OpenReview forum id `LeatkxrBCi`, https://openreview.net/forum?id=LeatkxrBCi). Both the forum page and `api2.openreview.net` returned a bot-challenge page, so decision type, scores and reviewer criticisms are **unknown**.
- Models (project page https://fuvty.github.io/C2C_Project_Page/): sharers Qwen2.5-0.5B, Llama3.2-1B, Qwen3-4B-Base; receiver Qwen3-0.6B; oracle study Qwen3-4B -> Qwen3-0.6B. Cross-family (Llama -> Qwen) is included.
- Datasets: not on the abstract or project page in this pass (my recollection is OpenBookQA, ARC-C, MMLU-Redux, C-Eval; unverified here).
- Baselines: receiver alone, text-to-text communication. Controls/ablations: "cache enrichment" oracle (62.3 vs 63.4), "cache transformation" (MLP projection), complementary-strength analysis.
- Seeds: not stated anywhere I could reach. Cost: 2.5x latency speedup vs text; no bytes/tokens/FLOPs on the pages fetched. Adapter size is only known via LCF's citation (956 MB).

**Latent Cache Flow (LCF), arXiv 2605.22863** — https://arxiv.org/abs/2605.22863
- Venue: a search snippet says "proceedings of the 43rd ICML, Seoul 2026", but the arXiv record says "6 pages, 5 figures" and the abstract calls its results "early experiments"; no OpenReview page found. Treat as **unverified, likely ICML 2026 workshop**.
- Models: not in abstract; built on C2C's Qwen pairs (inferred).
- Datasets: the "four benchmarks" of C2C (shared context) plus partitioned HotpotQA (different contexts).
- Baselines: C2C, text communication. Controls: none visible beyond baselines; a layer-pruning ablation.
- Seeds: unknown. Cost: **yes in bytes** — 13 MB adapter vs C2C 956 MB (24–76x smaller); 8.5x faster time-to-first-token; F1 +7.5, EM +23 on partitioned HotpotQA.

**Closed-form KV map, arXiv 2608.03893** — https://arxiv.org/abs/2608.03893 (NVIDIA)
- Venue: none listed; no OpenReview hit. **Unknown / preprint.**
- Models: six pairs in three families (Qwen3 14B->32B is the headline; others not on the abstract page). Constraint: "matched-KV pairs" (same head count and head dim).
- Datasets: calibration on 500 FineWeb-Edu sequences x 1,024 tokens; evaluation via HellaSwag accuracy retention (73–98% on four pairs, two pairs collapse).
- Baselines: re-prefill (upper bound); nonlinear MLP mapper recovers +37 pp on a failing pair. Controls: variance-explained analysis per source layer (56% key / 32% value); RoPE-stripped keys; multi-turn handoff stability.
- Seeds: unknown. Cost: 2.7–25x faster than re-prefill (time); not bytes/FLOPs.
- Third-party reimplementation exists: https://github.com/Susmith4710/kvtransfer.

**"Do latent channels actually communicate?", arXiv 2607.26773** — https://arxiv.org/abs/2607.26773
- Venue: **unknown**; no OpenReview hit.
- Models: Qwen3-4B, Qwen3-8B. Datasets: GSM8K, ARC-C, MATH-500.
- Baselines/controls: this paper *is* the control paper. Four message settings (real / replaced at the sender-receiver boundary) give five measurements: encoded sender information, receiver sensitivity to message presence and identity, task value of example-specific content, and value added by a separate agent. Headline: a -1.00 pp aggregate on GSM8K decomposes into -6.17 and +5.17 components.
- Seeds: unknown. Cost: not reported in the abstract.

Concurrent work surfaced by search, worth citing: "When Does Latent Communication Pay? A Causal Audit of Relayed KV Caches" (2608.04893), "Dual-Cache Latent Space Communication between Heterogeneous LMs" (2608.20617), CacheBridge (2609.00891), HyLaT hybrid latent-text protocol (2605.25421).

## (b) Checklist a strong paper in this space has

1. Receiver-alone and text-relay baselines on the same items (C2C).
2. Boundary-replacement controls: shuffled/deranged, content-free, wrong-example cache (2607.26773 sets the bar with four settings and five decomposed measurements).
3. Decomposition of the aggregate effect into "sender content" vs "any message present" (2607.26773).
4. Reported channel cost in bytes and in adapter parameters, next to accuracy (LCF: 13 MB vs 956 MB).
5. Latency or FLOPs vs re-prefill and vs text decode (closed-form paper: 2.7–25x; C2C/LCF: 2.5x / 8.5x TTFT).
6. Multiple model pairs including a cross-family pair and a failure case reported honestly (closed-form: six pairs, two collapse; C2C: Llama->Qwen).
7. An oracle / upper bound (C2C cache-enrichment oracle; closed-form's re-prefill).
8. Different-context setting, not just token-aligned shared context (LCF-X on partitioned HotpotQA).
9. Analysis of which layers carry the signal (closed-form's per-layer variance explained; LCF's layer pruning).
10. Linear vs nonlinear mapper comparison, so the claim "learned projector needed" is earned (closed-form paper).
11. Multiple benchmarks spanning knowledge and reasoning (2607.26773: GSM8K, ARC-C, MATH-500; C2C: four).
12. Seeds / confidence intervals — none of the four state seed counts on the pages I reached, so this is a bar you can set rather than meet.

## (c) Blunt gaps in our evidence

- One pair (3.7B -> 7B, same family). Every comparison paper has ≥3 pairs or a cross-family pair. The planned cross-family run is mandatory, not optional.
- No linear/closed-form baseline. 2608.03893 shows ridge regression recovers 73–98% within families; a reviewer will ask whether our learned projector beats a ridge map on the same pair. If it doesn't, the paper's contribution shifts to the controls and confidence work.
- Controls are good (deranged 5%, content-free 23% vs 73%) but the decomposition is not done: the 23% content-free number is exactly the "message present but wrong content" term that 2607.26773 isolates; report it that way and add a wrong-example (real cache from a different fact) control.
- No cost numbers. We must state projector parameter count, bytes per transferred cache vs bytes of the equivalent text, and prefill-skip latency. LCF's 13 MB is the reference point.
- One real dataset (SQuAD, F1 53 vs 55 reading text) and one synthetic. The "second dataset" should be a different-context or multi-hop task (HotpotQA partitioned, as LCF) because same-context SQuAD is the easy regime.
- Seeds and CIs: unknown for competitors, so three seeds with CIs on the synthetic task would exceed the field bar cheaply.
- Chance level 12.5% needs a clearer framing (8-way binding); reviewers will ask for a "receiver reads nothing" row alongside "receiver reads text".
- No layer analysis; which receiver layers accept the injected cache is cheap to report and both LCF and the closed-form paper do it.

## (d) ICML submission format (from ICML **2026** CFP; no 2027 CFP found on icml.cc)

- Main body 8 pages; references, impact statement, appendices unlimited; camera-ready gets one extra page (9). Source: https://icml.cc/Conferences/2026/CallForPapers and https://icml.cc/Conferences/2026/AuthorInstructions
- Style: `icml2026.zip` LaTeX style; US letter; `\usepackage[accepted]{icml2026}` for camera-ready; vector figures preferred.
- Impact statement: mandatory for the main track, separate section at the end before References, not counted toward the limit; boilerplate sentence permitted when impacts are routine.
- Reproducibility: no formal checklist in the 2026 CFP; "all claims must be … supported by reproducible experiments," code submission encouraged and "taken into account in the decision-making." Unlimited appendices go in the main submission file; no supplementary file at camera-ready.
- Double-blind; no public code links or acknowledgements in the submission.
- LLM writing assistance allowed with author responsibility; prompt injection is a desk-reject.
- 2026 dates for calibration: abstract Jan 23, full paper Jan 28 (AoE). Expect ICML 2027 to be similar; re-check icml.cc when the 2027 CFP posts.

Sources: [arXiv 2510.03215](https://arxiv.org/abs/2510.03215), [C2C project page](https://fuvty.github.io/C2C_Project_Page/), [OpenReview LeatkxrBCi (blocked)](https://openreview.net/forum?id=LeatkxrBCi), [arXiv 2605.22863](https://arxiv.org/abs/2605.22863), [arXiv 2608.03893](https://arxiv.org/abs/2608.03893), [kvtransfer reimplementation](https://github.com/Susmith4710/kvtransfer), [arXiv 2607.26773](https://arxiv.org/abs/2607.26773), [ICML 2026 CFP](https://icml.cc/Conferences/2026/CallForPapers), [ICML 2026 Author Instructions](https://icml.cc/Conferences/2026/AuthorInstructions).
