# Latent communication between LMs: survey notes (2026-09-12)

Compiled by a research agent on 2026-09-12 for the K2 Cascade follow-up. Entries marked (mem) were not re-verified that day.

## What works
1. Same instance, training-free, KV / last-layer hidden state as prefix: LatentMAS (arXiv 2511.20639, Qwen3 4/8/14B; up to +14.6%, −70–84% tokens, 4x faster; github.com/Gen-Verse/LatentMAS), KVComm (2510.12872; TTFT 430→55 ms).
2. Same-arch KV reuse with selective layer recompute: DroidSpeak (2411.02820; 3.1x prefill).
3. Cross-size KV with a small trained gated residual adapter and LM loss: C2C (2510.03215, ICLR'26, github.com/thu-nics/C2C; +3–5% vs text, 2–2.5x faster, adapter 956 MB), Latent Cache Flow (2605.22863; 13–107 MB adapter, 300 steps, 4.5 h on one A100, 500k OpenHermes; +7.5 F1), XKV (2608.20617), dense latent comm on Qwen3 4/8/14B (2606.13594; ≥ text at 2–3x lower compute).
4. Same-family closed-form ridge KV map (NVIDIA, 2608.03893): strip RoPE, per-head ridge from top-k source layers, 500×1024 FineWeb-Edu seqs; retention 97.6% (Qwen3 14→32B), 87.5%, 72.8% (Llama 8→70B), 44% (Ministral 3→14B); MLP adds up to +37 pp on bad pairs.
5. Draft heads consuming target features: EAGLE-3 (2503.01840), HASS.
6. Latent CoT inside one model needs SFT/curriculum (Coconut 2412.06769, CODI 2502.21074); training-free Soft Thinking (2505.15778) ≈ +2.5 pp.

## What fails
- Training-free hidden-state add/replace across sizes (Pythia 160M→410M, 2606.03280): cos 0.97 alignment, zero downstream gain; replace is destructive.
- Reconstruction-only objectives (Dery 2601.06123; NVIDIA off-distribution R² < 0).
- Unrepaired position-shifted caches are worse than no cache (KVShareArena 2609.10266).
- Latent channels are attackable without integrity checks (2606.28958, 2605.28214).

## Gap
No published work where the transferred KV/hidden state carries a receiver-usable "I can't do this" signal. 2606.20662 argues uncertainty is laundered at handoff (text summaries too) but reports no experiment.

## Plan for K2 (3.7B → 7B, same tokenizer)
Baselines: (a) 7B re-reads the full text (upper bound); (b) 3.7B writes a text summary for the 7B at matched byte budget.
Metrics: continuation-NLL retention = (NLL_none − NLL_latent) / (NLL_none − NLL_full); next-action agreement and task pass on held-out trajectories; TTFT; bytes transferred.
1. Closed-form ridge KV map fitted on our agent trajectories (one A100-hour). Falsified if retention < 50%.
2. LCF-style gated residual adapter trained with suffix-LM loss on trajectories (300 steps, ~4.5 h). Falsified if it never beats (b) at equal bytes.
3. Confidence-in-latent: probe output (p(step fails)) passed as an extra soft token to the 7B. Only if the pre-action probe AUROC > 0.7.
Differentiation from 2606.13594: agent trajectories instead of generic text, and the self-knowledge signal.
