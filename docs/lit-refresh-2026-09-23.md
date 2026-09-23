*Research-agent pass, 2026-09-23: arXiv 2026-08-10 → 09-23. No derange/content-free control or sender-uncertainty test found elsewhere; nearest overlap 2608.30963 and 2609.25053.*

Tool budget used (15). Findings below.

## Summary

Five new papers in the 2026-08-10 to 09-23 window touch cross-model latent/KV handoff; none reports a derange/content-free control or a sender-uncertainty transfer test, so none scoops the 3.7B->7B paper's core claim. The closest threat is 2608.30963 (cross-family KV sharing incl. 70B->7B), and the closest methodological neighbor is StateBridge 2608.13317 (training-free, COLM 2026).

## New papers (not on the known list)

**1. arXiv:2608.30963 - "A Universal Context-Reuse Layer for Cross-Model KV Sharing"** (Li, Jiang, Zhao, Li; 2026-08-31, v1)
- Claim: a translation layer that makes a source model's KV state consumable by a target differing in scale, architecture, attention config, tokenizer, family.
- Models: Qwen2.5-7B -> 1.5B (within-family, downward); Qwen2.5-1.5B -> Gemma-2-2B (cross-family); Llama3.1-70B -> Qwen2.5-7B (cross-family, cross-size). LongBench2 27.59% -> 34.48% on the 7B->1.5B pair; 70B->7B gets 44.0 vs 45.7 native at 138ms vs 899ms prefill.
- Relation to us: broadest coverage of any paper in the window (tokenizer mismatch, cross-family). Abstract has no SQuAD/QA-F1, no shuffle/content-free control, no uncertainty transfer. This is the paper a reviewer will ask us to compare against; the answer is that our contribution is the causal-control and uncertainty-transfer evidence, not another mapper. Flag: partial overlap on the "cross-size KV projection works" headline.

**2. arXiv:2608.13317 - "StateBridge: Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems"** (Peng, Zhang, Wang, Aletras; 2026-08-13, v1; COLM 2026)
- Claim: closed-form orthogonal map from sender final-layer hidden states to receiver input-embedding space, plus norm calibration and vocabulary anchoring; no training.
- Models: four models across two families (sizes not in abstract; unverified). Best or tied-best on 22/26 model-task pairs (math, code, QA).
- Cross-size and cross-family: yes.
- Relation: transfers final-layer hidden states into the input space, not per-layer KV into the receiver's cache, so mechanism differs from ours. No derange/content-free control mentioned in abstract; no confidence transfer. Its training-free result is a useful baseline to cite against our fitted projector.

**3. arXiv:2609.25053 - "LatentPort: Beyond KV Cache - Cross-Model Transfer of Recurrent Memory in Hybrid Language Models"** (Villani; 2026-09-06/07, v1)
- Claim: 4B -> 9B Qwen3.5 (hybrid GDN + attention) handoff of attention KV plus recurrent/conv state without target prefix replay; -0.747 nats/token teacher-forced NLL, JSD 0.022, 0.918 native-context recovery on PG19 + web docs.
- Cross-size, same family, architecture-matched.
- Relation: same "small sender -> larger receiver, no replay" shape as ours, but evaluated on continuation NLL, not QA; single pair, single direction; author explicitly says free-generation equivalence unproven. Controls are ablations (KV-only, learned-map vs direct copy), not derange/content-free. No uncertainty. Not a scoop, but cite it as concurrent evidence that upward transfer is feasible.

**4. arXiv:2608.20927 - "MentorPulse: Refreshing Cross-Model Latent Guidance for Long-Form Generation"** (Liu, Li, Qiu, Kong, Kalnis; 2026-08-21, v1)
- Claim: static C2C-style guidance hurts a 4B student on long-form generation (-2.5 pts); refreshing compressed mentor states every 16 tokens through slot memory + gated cross-attention closes 52.2% of the mentor-student gap. Baselines: C2C, T2T, equal-budget LoRA.
- Models: 11 mentor-student pairs across three families; student 4B (mentor sizes unverified). Cross-size and cross-family.
- Relation: large -> small direction (opposite of ours); learned cross-attention rather than cache injection. Relevant negative result for us: a one-shot latent handoff can degrade below no-guidance on long outputs. Our SQuAD short-answer setting avoids this, but worth noting as a scope limit.

**5. arXiv:2609.11365 - "Portable Semantics, Private Dialects: Reuse and Negative Transfer in Latent Communication Between Language-Model Cells"** (Marincat; 2026-09-10, v1)
- Claim: independently trained latent-communication interfaces fail zero-shot transfer across societies (0.169 -> 0.857 accuracy only after reinitializing interface components); preregistered raw/orthogonal/linear/nonlinear alignment ladder over 30 ordered pairs.
- Models: not in abstract (unverified); small trained "cells," apparently not pretrained LLMs.
- Relation: not KV transfer between pretrained models, but its "alignment ladder" and leakage-controlled causal audit is close in spirit to our derange/content-free controls. Cite for methodology only.

## Tangential (checked, not in scope)

- 2609.24197 H-Spec (2026-09-21): speculative decoding that reuses target KV plus last-position target hidden state for the drafter; same-model target state, not cross-model. Not a scoop.
- 2609.14717 Carryover Drafting: recycles rejected drafter states; within one spec-decoding pair. Not cross-model communication.
- 2609.06940 Unified AI Gateway (2026-09-06): joint routing + KV cache management for serving; title suggests infrastructure, not cross-model cache translation. Unverified, abstract not fetched.
- 2608.19161 covert coordination in latent MAS (2026-08-19): safety/detection angle on latent channels; unverified.
- 2608.10198 post-hoc sparse coding of latent comms between VLM agents (2026-08-10, boundary date): interpretability; unverified.

## Known-list papers: version checks

- 2606.05711 (Beyond Tokens survey): v3 dated 2026-07-15, before the window; no new version.
- 2608.03893, 2608.20617, 2609.00891 appear in the arXiv listing with their original dates only; no new versions observed (search listing shows announce date only, so a quiet v2 is possible; unverified).
- No new versions seen for the other known ids; I did not fetch each abs page individually.

## Scoop / contradiction verdict

- No paper in the window reports (a) derange or content-free cache controls, (b) SQuAD F1 near the text ceiling for a KV projector, or (c) a sender-uncertainty transfer test. The uncertainty-in-latent-handoff question appears unclaimed as of 2026-09-23.
- Nearest overlap on the headline "cross-size KV projection works": 2608.30963 (broader model coverage) and 2609.25053 (same upward direction, hybrid arch). Neither contradicts our results.
- One caution: MentorPulse's finding that static latent guidance can underperform no-guidance on long-form generation is a limit we should preempt in the discussion.

## Coverage gaps

Fetched only page 1 of the "cross-model KV cache" listing (36 of 191 results shown, sorted by date, so the window is covered but truncated entries 37-50 were not seen). Did not run a separate "hidden state injection" or "activation communication" query; a second pass on those phrasings would tighten the sweep.

Sources: [2608.30963](https://arxiv.org/abs/2608.30963), [2608.13317](https://arxiv.org/abs/2608.13317), [2609.25053](https://arxiv.org/abs/2609.25053), [2608.20927](https://arxiv.org/abs/2608.20927), [2609.11365](https://arxiv.org/abs/2609.11365), [2609.24197](https://arxiv.org/abs/2609.24197), [2609.14717](https://arxiv.org/pdf/2609.14717), [2609.06940](https://arxiv.org/pdf/2609.06940), [2606.05711](https://arxiv.org/abs/2606.05711), [2609.00891](https://arxiv.org/abs/2609.00891), [2608.03893](https://arxiv.org/abs/2608.03893), [arXiv search: cross-model KV cache](https://arxiv.org/search/?query=cross-model+KV+cache&searchtype=all&order=-announced_date_first&size=50), [arXiv search: latent communication](https://arxiv.org/search/?query=latent+communication+language+models&searchtype=all&order=-announced_date_first&size=50)
