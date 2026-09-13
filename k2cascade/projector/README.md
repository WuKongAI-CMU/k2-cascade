# KV-cache projector: K2-Horizon-3.7B -> K2-Horizon-7B

Train a small projector that turns the 3.7B's KV cache of a prefix into a KV cache the 7B can read, so
the 7B skips prefill on context the 3.7B already processed. Both models frozen; only the projector trains.

Facts (from the HF `config.json` of both models, fetched 2026-09-12):

| | 3.7B (source) | 7B (target) |
|---|---|---|
| layers | 36 | 36 |
| hidden | 2560 | 4096 |
| q-heads / kv-heads / head_dim | 32 / 8 / 128 | 32 / 8 / 128 |
| per-layer K or V of a prefix S | (B, 8, S, 128) | (B, 8, S, 128) |
| RMSNorm groups (`layernorm_num_groups`) | 2 | 4 |
| rope theta / rope_head_dim | 1e7 / 128 (full RoPE) | 1e7 / 128 |
| `query_key_norm` / attention gate / sliding window | no / no / no | no / no / no |
| vocab / chat template | 250,624 shared | same |
| bf16 safetensors | ~10 GB | 18.0 GB |

Same depth, same KV shape ("matched-KV pair" in arXiv 2608.03893): the default layer map is identity and
the default ridge is per kv head (8 maps of 256 -> 256 per layer).

## Files

| file | what |
|---|---|
| `extract.py` | forward hooks on `model.model.layers[i].self_attn.k_proj` / `.v_proj` -> **pre-RoPE** K, V per layer (B, 8, S, 128); hooks on the decoder layers -> residual stream. RoPE helpers `rope_cos_sin / apply_rope / strip_rope`; `make_cache` applies RoPE and builds a `DynamicCache` for injection (autograd flows through). |
| `ridge.py` | closed-form ridge per target layer (and per head), streaming Gram accumulation, `last_aligned` or CKA `topk` layer map, save/load. CLI fits + evaluates. |
| `mlp.py` | LCF-style projector: `ridge(x) + sigmoid(alpha) * up(SwiGLU(down(x)))`, bottleneck `d`, per-head alpha; step 0 == ridge. |
| `train.py` | frozen models, bf16 autocast, gradient checkpointing (`use_reentrant=False`), AdamW 1e-4 / wd 0.01 / 10% warmup + cosine, grad accumulation, `--steps` cap, checkpoint + `--resume`, optional wandb. |
| `eval.py` | retention metric (below), chunked LM head over continuation positions only, `messages_continuation_loss` hook for agent traces. |
| `data/prepare_fineweb.py` | streams `HuggingFaceFW/fineweb-edu` sample-10BT, writes N packed 1024-token sequences as JSONL. |

Metric (`eval.py`): with a 512-token prefix and its 512-token continuation, continuation NLL of the 7B under
three arms: `none` (no prefix), `oracle` (7B prefills the prefix itself), `project` (cache mapped from the 3.7B).
`retention = (none - project) / (none - oracle)`. The CLI also prints `source_oracle` (the 3.7B reading its own
prefix), which is what the small model alone achieves. 1024/1024 as in the paper: `--seq_len 2048 --prefix_len 1024`.

## A100 40 GB run (GCP `a2-highgpu-1g`, us-central1: on-demand $3.673/hr; spot reported $1.10-2.12/hr, unverified)

```bash
# 0. environment (python 3.12; torch cu12x; flash-attn optional)
sudo apt-get install -y git curl && curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <this repo> && cd k2-cascade
uv sync --extra data   # if mlx-lm (a base dep for the laptop) fails to resolve on Linux: uv venv && uv pip install torch transformers accelerate safetensors datasets
# torch from uv sync is the default PyPI wheel; make sure it is the CUDA build:
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
uv pip install flash-attn --no-build-isolation         # optional; then pass --attn flash_attention_2
uv run python -c "import torch, transformers; print(torch.cuda.get_device_name(), transformers.__version__)"
huggingface-cli download IFM/K2-Horizon-3.7B && huggingface-cli download IFM/K2-Horizon-7B   # ~28 GB

# 1. data: 2048 sequences of 1024 tokens (32 held out by default), ~5 min
uv run python -m k2cascade.projector.data.prepare_fineweb --n 2048 --out data/fineweb_1024.jsonl

# 2. ridge baseline (per head, identity map, 64 sequences; paper uses 500) -> runs/ridge, prints retention
uv run python -m k2cascade.projector.ridge --data data/fineweb_1024.jsonl --out runs/ridge --seqs 64
uv run python -m k2cascade.projector.ridge --data data/fineweb_1024.jsonl --out runs/ridge_k2 --seqs 256 --k 2 --map topk
uv run python -m k2cascade.projector.ridge --data data/fineweb_1024.jsonl --out runs/ridge_flat --seqs 256 --flat

# 3. MLP projector on top of the best ridge (LCF recipe: 300 steps, effective batch 16 here)
uv run python -m k2cascade.projector.train --ridge runs/ridge --data data/fineweb_1024.jsonl \
    --out runs/mlp --steps 300 --batch 2 --accum 8 --seq_len 1024 --prefix_len 512 --bottleneck 128
#    resume after a preemption:  add --resume (reads runs/mlp/last.pt)
#    wandb:                      add --wandb   (needs WANDB_API_KEY)

# 4. eval any saved projector
uv run python -m k2cascade.projector.eval --projector runs/mlp/step_300 --data data/fineweb_1024.jsonl
uv run python -m k2cascade.projector.eval --projector runs/ridge --data data/fineweb_1024.jsonl
```

Four-hour budget (estimates, not measured; check `s_per_step` in the log after 5 steps and rescale):
model download 10 min; data 5 min; each ridge fit 3-10 min (two forwards per sequence, no backward);
MLP step = 16 sequences x (3.7B forward + 7B forward/backward with recompute) ~ 5-8 s, so 300 steps ~ 30-40 min.
Order: ridge (3 variants, ~30 min) -> MLP 300 steps -> eval -> if > 1.5 h remain, `--steps 1000 --out runs/mlp_long`.
Stop-loss: if the ridge `retention` is below 0.3 on the first try, spend the time on `--k 2 --map topk` and
`--lam` sweeps (1e-3 .. 1e-1) before any MLP run; the MLP only adds on top of the ridge.

## Memory budget (bf16 weights, batch 2 x 1024, prefix 512)

| item | GB |
|---|---|
| 7B weights (18.0 GB safetensors) | 18.0 |
| 3.7B weights | 10.1 |
| projector fp32 params + grads + Adam (per-head, d=128: ~66M params) | 1.1 (flat layout d=128: ~177M params, 2.8) |
| injected cache with grad, 36 layers x K,V x (2, 8, 512, 128) bf16 | 0.15 (+0.15 grad) |
| 7B activations with checkpointing (one layer live + 36 saved inputs) | ~1.5 |
| LM head, chunked 1024 tokens x 250,624 vocab, fp32 logits + grad | ~2.0 |
| ridge Gram accumulators (per-head 36x8x257^2 fp32 / flat 36x2049^2) | 0.08 / 0.6 |
| **total** | **~33 GB** (batch 4: ~36 GB) |

Never materialise full logits for the whole sequence: 2 x 1024 x 250,624 fp32 = 2 GB per copy, plus grad.
`eval.continuation_loss` applies `lm_head` only to continuation positions, 1024 tokens at a time under
`torch.utils.checkpoint`.

## What to expect

- arXiv 2608.03893 (closed-form ridge, per head, RoPE-stripped keys, 500 FineWeb-Edu sequences of 1,024 tokens):
  one source layer explains 56% of target key variance and 32% of value variance on Qwen3 14B->32B (79% / 65%
  with several source layers); the linear mapper retains 73-98% of standalone-prefill accuracy on four of six
  pairs, two pairs degrade sharply, and an MLP recovers up to +37 pp HellaSwag retention on the failures.
  Their retention is on downstream accuracy; ours is on continuation NLL, so expect the NLL number to be lower
  than their accuracy number for the same mapper.
- arXiv 2605.22863 (Latent Cache Flow): joint K/V translation through a bottleneck gives an adapter ~4% of C2C's
  size (13 MB pruned vs 956 MB) and beats C2C in shared-context settings; recipe lr 1e-4, wd 0.01, ~300 steps.
- Our guess for 3.7B -> 7B: ridge retention 0.4-0.8 (same depth and KV shape, but hidden 2560 vs 4096 and
  different norm grouping); MLP +0.05-0.15 on top. Below 0.3 means the layer map or lambda is wrong, not the idea.

## Hook points and assumptions to check on the GPU (fix fast if they differ)

1. `model.model.layers[i].self_attn` is `K2HorizonAttention` with `k_proj`, `v_proj` (`extract.attention_modules`;
   falls back to any module owning both). Verified on the vendored remote code (`vendor/k2_horizon`, fetched
   2026-09-12 from IFM/K2-Horizon-7B); `trust_remote_code=True` loads the same file.
2. `k_proj` output is pre-RoPE and the cache stores post-RoPE keys + raw values: `tests/projector/test_extract.py`
   checks `apply_rope(k_proj out) == past_key_values.layers[i].keys` on a tiny random model.
3. The remote code ignores `output_hidden_states`, so hidden states come from hooks on `model.model.layers[i]`.
4. `DynamicCache(config=...)` + `cache.update(K, V, i)` per layer; `continuation_loss` passes `position_ids`
   and `cache_position = arange(P, P+C)`. No padding anywhere, so no attention mask is built by hand.
5. `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` then `.train()`
   on the frozen target (HF only checkpoints in train mode; dropout is 0). Verified to enable on the tiny model;
   the actual backward through checkpointed layers with an injected cache is only exercised on CPU in tests
   with `--no-grad_ckpt`. If it errors on CUDA, run with `--no-grad_ckpt --batch 1 --accum 16`.
6. bf16: models load in bf16 on CUDA; the projector keeps fp32 params under `torch.autocast(bfloat16)`.
   Not run here (CPU only). If loss is NaN, try `--no-bf16` on the projector side (models stay bf16).
7. Loads use `dtype=` (transformers 5 name); `attn_implementation` default `sdpa`.

## Tests

`uv run pytest -q tests/projector` (9 tests, ~1 s, CPU, tiny random K2 models from the vendored classes):
hook shapes + pre-RoPE check, RoPE strip/re-apply round trip and match with the model's rotary module,
injected own-cache == own prefill logits, ridge fit + inject on a memorizing toy (retention > 0.3; measured 0.99),
train 6 steps with decreasing finite loss, checkpoint save/resume, eval-arm consistency.
