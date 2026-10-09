import torch

from k2cascade.projector.noma import Encoder, make_episodes
from k2cascade.projector.workspace import ARMS, Workspace, band, coords, swap
from tests.projector.test_noma import FakeTok, identity_projector


def _ws(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = model.config.hidden_size
    return Workspace(model, {0: torch.eye(d) + 0.1 * torch.randn(d, d, generator=g)})


def test_decompose_is_sparse_and_exact(target):
    ws = _ws(target)
    h = torch.randn(4, target.config.hidden_size)
    ids, w, jp = ws.decompose(0, h, k=5)
    assert ids.shape == (4, 5) and (w >= 0).all()
    recon = sum(w[:, i, None] * ws.atoms(0, ids[:, i]) / ws.norms(0)[ids[:, i]][:, None] for i in range(5))
    assert torch.allclose(recon, jp, atol=1e-4) and (h - jp).norm() < h.norm()


def test_swap_and_coords(target):
    enc = Encoder(FakeTok(), 3, 4)
    eps = make_episodes(4, 3, 4, seed=2)
    ws = _ws(target)
    r = swap(target, target, identity_projector(target), enc, eps, ws, [0], k=4)
    assert set(ARMS) <= set(r) and 0 <= r["full"]["follow"] <= 1 and 0 < r["j_share_mean"] <= 1
    c = coords(target, target, enc, eps, ws, ws, k=3, n_pages=2)
    assert abs(c["j"][0][0] - 1.0) < 1e-6 and c["j_shuffled"][0][0] <= 1.0
    assert band(36) == [l for l in range(35) if 0.38 <= l / 35 <= 0.92]
