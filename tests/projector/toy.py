"""Tiny random-weight K2 Horizon models built from the vendored remote code (vendor/k2_horizon)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))

from k2_horizon.configuration_k2_horizon import K2HorizonConfig  # noqa: E402
from k2_horizon.modeling_k2_horizon import K2HorizonForCausalLM  # noqa: E402

VOCAB = 64


def tiny_k2(hidden: int = 32, layers: int = 2, n_kv: int = 2, head_dim: int = 8, groups: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    cfg = K2HorizonConfig(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
        num_attention_heads=2 * n_kv, num_key_value_heads=n_kv, head_dim=head_dim, rope_head_dim=head_dim,
        query_key_norm=False, layernorm_num_groups=groups, num_experts=0, num_experts_per_tok=0,
        moe_intermediate_size=0, mlp_only_layers=list(range(layers)), mova_num_experts=0,
        rope_parameters={"rope_theta": 1e7, "rope_type": "default"}, max_position_embeddings=512,
        use_sliding_window=False, sliding_window=None, tie_word_embeddings=False)
    return K2HorizonForCausalLM(cfg).eval()
