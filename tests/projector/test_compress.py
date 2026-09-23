import torch

from k2cascade.projector.compress import apply


def _kv(L=4, B=1, H=2, T=6, d=8):
    g = torch.Generator().manual_seed(0)
    return [torch.randn(B, H, T, d, generator=g) for _ in range(L)], [torch.randn(B, H, T, d, generator=g) for _ in range(L)]


def test_none_is_identity_and_reports_full_size():
    k, v = _kv()
    k2, v2, info = apply(k, v, None)
    assert all(torch.equal(a, b) for a, b in zip(k, k2)) and info["bytes_per_token"] == info["bytes_per_token_full"]


def test_layers_heads_rank_quant():
    k, v = _kv()
    k2, v2, info = apply(k, v, "layers=2-3,heads=1,int8")
    assert torch.count_nonzero(k2[0]) == 0 and torch.count_nonzero(k2[2][:, 1]) == 0 and torch.count_nonzero(k2[2][:, 0]) > 0
    assert info["layers"] == 2 and info["heads"] == 1 and info["bits"] == 8
    assert info["bytes_per_token"] == 2 * 2 * 1 * 8 * 8 / 8
    assert (k2[2][:, 0] - k[2][:, 0]).abs().max() < 0.05  # int8 is close
    k3, _, _ = apply(k, v, "rank=1")
    assert torch.linalg.matrix_rank(k3[0][0, 0]) == 1
    k4, _, _ = apply(k, v, "int4")
    assert (k4[0] - k[0]).abs().max() < 0.6
