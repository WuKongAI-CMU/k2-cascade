import torch

from k2cascade.projector.extract import (apply_rope, extract, kv_geometry, make_cache, rope_cos_sin,
                                         rope_theta, strip_rope)
from tests.projector.toy import VOCAB


def test_hooks_capture_pre_rope_kv_with_right_shapes(target):
    ids = torch.randint(0, VOCAB, (2, 10))
    b = extract(target, ids)
    n_kv, d = kv_geometry(target.config)
    L = target.config.num_hidden_layers
    assert b.num_layers == L and len(b.values) == L and len(b.hidden) == L + 1
    for k, v in zip(b.keys, b.values):
        assert k.shape == (2, n_kv, 10, d) and v.shape == (2, n_kv, 10, d)
    assert b.hidden[0].shape == (2, 10, target.config.hidden_size)
    # the model's own cache holds post-RoPE keys and raw values: rotating our pre-RoPE keys must match it
    with torch.no_grad():
        pkv = target(input_ids=ids, use_cache=True).past_key_values
    pos = torch.arange(10)[None].expand(2, -1)
    cos, sin = rope_cos_sin(pos, d, rope_theta(target.config))
    for i in range(L):
        torch.testing.assert_close(apply_rope(b.keys[i], cos, sin), pkv.layers[i].keys, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(b.values[i], pkv.layers[i].values, atol=1e-5, rtol=1e-4)
    assert not torch.allclose(b.keys[0], pkv.layers[0].keys)  # i.e. we really captured pre-RoPE


def test_rope_strip_reapply_round_trip(target):
    x = torch.randn(2, 2, 7, 8)
    pos = torch.tensor([[0, 1, 2, 3, 100, 1000, 5000]] * 2)
    cos, sin = rope_cos_sin(pos, 8, 1e7)
    torch.testing.assert_close(strip_rope(apply_rope(x, cos, sin), cos, sin), x, atol=1e-5, rtol=1e-5)
    # matches the model's own rotary module
    mcos, msin = target.model.rotary_emb(x, pos)
    torch.testing.assert_close(mcos, cos, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(msin, sin, atol=1e-6, rtol=1e-6)


def test_make_cache_reproduces_oracle_logits(target):
    """Injecting the target's own pre-RoPE KV through make_cache == the target's own prefill."""
    ids = torch.randint(0, VOCAB, (1, 12))
    prefix, cont = ids[:, :8], ids[:, 8:]
    with torch.no_grad():
        b = extract(target, prefix, with_hidden=False)
        cache = make_cache(target, b.keys, b.values)
        pos = torch.arange(8, 12)
        inj = target(input_ids=cont, past_key_values=cache, position_ids=pos[None], cache_position=pos).logits
        full = target(input_ids=ids, use_cache=False).logits[:, 8:]
    torch.testing.assert_close(inj, full, atol=1e-4, rtol=1e-4)
    assert cache.get_seq_length() == 12
