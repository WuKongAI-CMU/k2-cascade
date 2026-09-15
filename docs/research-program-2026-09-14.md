# Research program notes, 2026-09-14

Deep-dive follow-up to `latent-communication-survey-2026-09-12.md`. Four questions, answered.

## 1. Is "confidence in the latent" actually unclaimed?

Narrower than we thought. **Confidence Laundering in Agent Systems: Why Uncertainty Needs a Latent Carrier**
([arXiv 2606.20662](https://arxiv.org/abs/2606.20662)) stakes exactly this framing. Its execution is thin:
Qwen-1.5B only, HotpotQA + Tavily (one search-then-answer hop, not multi-step), the "downstream consumer" is a
logistic-regression probe rather than a second model, ordering reported in a figure with no numbers, no cross-size pair.
Treat it as a position paper to cite and to beat on execution.

Everything else passes a *decision*, not a continuous signal: Gupta et al. ([2404.10136](https://arxiv.org/abs/2404.10136))
and Gatekeeper ([2502.19335](https://arxiv.org/abs/2502.19335)) threshold a scalar and the large model then re-prefills the
original prompt. Bayesian Self-Escalation ([2608.24087](https://arxiv.org/html/2608.24087)) escalates mid-generation on a
matched pair (Qwen2.5-Coder 1.5B→7B, MBPP, 257 cases) but explicitly transmits text artifacts, "not raw hidden-state activations."
The sender-side signal is known to exist and be linear: **Decomposing and Steering Functional Metacognition**
([2605.08942](https://arxiv.org/abs/2605.08942)) finds "self-assessed capability" linearly decodable from the residual
stream and causally steerable — but never transferred between models.

**Open slot: a continuous uncertainty-bearing latent consumed by a different, larger model.**

## 2. How to evaluate on agent steps

Replay is disqualified. **The Replay Gap** ([2608.08239](https://arxiv.org/abs/2608.08239), COLM 2026): mini-SWE-agent,
Qwen3-4B vs 14B, SWE-bench Verified, 717 scored branch pairs. Mid-trajectory model swaps rewrote 61–94% of post-fork
actions and left only 3.2–39.4% of replayed states valid (controls 73.8–85.8%); a log-stitching replay evaluator went
0-for-5 on success-relevant calls. So: **branched rollouts with matched same-model control forks**, fork position and
swap direction as variables.

Field practice: SWE-Router ([2607.00053](https://arxiv.org/abs/2607.00053)) uses repository-disjoint splits, Route-AUC
(cost vs resolve), and Monte-Carlo permutation bands. Split and bootstrap **by trajectory, not by step**.

Nobody yet evaluates a KV/hidden-state handoff *inside* a real agent loop: 2608.03893 and CacheBridge
([2609.00891](https://arxiv.org/abs/2609.00891)) motivate agentic switching but report HellaSwag-style retention only.

## 3. Will 3.7B→7B transfer?

NVIDIA's own retention numbers ([2608.03893](https://arxiv.org/html/2608.03893v1)): Qwen3 14B→32B 97.6%, Qwen3 8B→32B 87.5%,
Ministral 3B→8B 76.2%, Llama 3.1 8B→70B 72.8%, but Ministral 3B→14B 44.2% and 8B→14B 41.6%. Attention-output cosine
correlates with retention (r=+0.57); calibration R² does not (r=−0.20). Crucially the cosine diagnostic is **post-hoc** —
it requires fitting the mapper first. So matched-KV geometry is encouraging, not a guarantee, and the one-GPU-hour ridge
baseline is the right first move.

Theory: Representation Alignment Rests on Linear Structure ([2605.28870](https://arxiv.org/pdf/2605.28870)) splits
alignment into signal / bias / noise; Back into Plato's Cave ([2604.18572](https://arxiv.org/abs/2604.18572)) shows
Platonic-convergence evidence degrading at scale. Relative representations ([2209.15430](https://arxiv.org/pdf/2209.15430))
are the anchor-based fallback if a direct linear map underperforms.

## 4. What reviewers demanded from accepted papers

- **StateBridge** (COLM 2026, [2608.13317](https://arxiv.org/abs/2608.13317)): a reviewer wrote *"the paper does not
  demonstrate latent communication between different models"*; rebuttal forced a token-embedding-only prefix baseline, a
  raw-unaligned-prefix baseline (which collapsed), 3 seeds, and end-to-end wall-clock — which showed it **slower than text**
  (162 s vs 110 s) and cost it points.
- **ICaRus** (ICLR 2026): *"How does the accuracy compare with not sharing at all?"*
- **RelayCaching** (ICML 2026): *"A critical baseline is missing: RoPE re-application alone, without rectification."*
- **KaVa** (ICLR 2026): dinged for 0.5B–3B only and GSM8K only.
- **What Do Latent Agents Actually Represent?** (ICML 2026): correctness signals in latent channels *"appear equally in
  single-agent baselines"* — a mandatory control.

## The claim we would defend

**Escalation Latents: Passing a Small Model's Own Uncertainty Through the KV Cache Improves Mid-Trajectory Handoff.**
In a matched-KV same-family pair (K2 Horizon 3.7B→7B), a map from the small model's KV state into the large model's cache
can be trained so the transferred latent carries not only context but a decodable "this step will be rejected" direction;
the receiver conditioned on it resolves more agent steps per dollar than (i) text handoff, (ii) scalar-confidence deferral,
and (iii) the same latent with the uncertainty direction projected out. Arm (iii) is the paper.

Minimum experiments: probe with trajectory-level splits and a metadata-shortcut baseline; ridge KV map with retention;
branched-rollout agent eval with matched control forks; the four arms; the ablations reviewers always demand.

Two ways it dies: the probe turns out to be a shortcut (see `k2cascade/shortcut.py`), or the projector plus branched
inference costs more wall-clock than just calling the big model.

## Deadlines
ICLR 2027 abstract **2026-09-18**, paper **2026-09-25**. NeurIPS 2026 closed 2026-05-06. ARR cycle 2026-10-12.
ICML 2027 / COLM 2027 not announced.
