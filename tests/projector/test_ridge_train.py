import torch

from k2cascade.projector.extract import extract
from k2cascade.projector.eval import continuation_loss, evaluate, oracle_cache, projected_cache
from k2cascade.projector.mlp import MLPProjector
from k2cascade.projector.ridge import RidgeProjector, fit_ridge, last_aligned_map, layer_correlation, topk_map
from k2cascade.projector.train import TrainConfig, batches, load_checkpoint, make_optimizer, train
from tests.projector.toy import VOCAB


def test_layer_maps():
    assert last_aligned_map(36, 36) == [[i] for i in range(36)]
    assert last_aligned_map(4, 6, k=2) == [[0], [0], [0], [0, 1], [1, 2], [2, 3]]


def _fit_batches(seq, n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [seq] + [torch.randint(0, VOCAB, (1, 48), generator=g) for _ in range(n)]


def test_ridge_fit_inject_recovers_prefix_information(source, memorized):
    """Target memorized one sequence; ridge-mapped *source* KV of the prefix must lower the target's
    continuation loss relative to no prefix, i.e. retention > 0 (and loss(project) < loss(none))."""
    tgt, seq = memorized
    for per_head in (True, False):
        ridge = fit_ridge(source, tgt, _fit_batches(seq), lam=1e-3, per_head=per_head)
        assert ridge.W.shape[0] == 2 and ridge.per_head == per_head
        res = evaluate(source, tgt, ridge, [seq], prefix_len=32)
        assert res["oracle"] < 0.5 * res["none"], res  # memorization worked: prefix matters
        assert res["project"] < res["none"], res
        assert res["retention"] > 0.3, res
    # topk map via CKA gives a valid map too
    corr = layer_correlation(extract(source, seq), extract(tgt, seq))
    assert corr.shape == (2, 2) and all(len(m) == 1 for m in topk_map(corr, 1))


def test_ridge_save_load_roundtrip(source, target, tmp_path):
    ridge = fit_ridge(source, target, _fit_batches(torch.randint(0, VOCAB, (1, 48)), n=4))
    ridge.save(tmp_path / "r")
    back = RidgeProjector.load(tmp_path / "r")
    torch.testing.assert_close(back.W, ridge.W)
    assert back.layer_map == ridge.layer_map and back.per_head


def _toy_cfg(tmp_path, **kw):
    base = dict(out=str(tmp_path / "run"), steps=6, batch=2, accum=1, seq_len=48, prefix_len=32, lr=1e-2,
                warmup=0.0, bottleneck=16, bf16=False, grad_ckpt=False, save_every=3, log_every=100)
    base.update(kw)
    return TrainConfig(**base)


def test_train_runs_and_loss_decreases(source, memorized, tmp_path):
    tgt, seq = memorized
    ridge = fit_ridge(source, tgt, _fit_batches(seq), lam=1e-3)
    proj = MLPProjector.from_ridge(ridge, bottleneck=16)
    assert torch.allclose(proj.gates(), torch.full_like(proj.gates(), torch.sigmoid(torch.tensor(-4.0)).item()))
    with torch.no_grad():  # step 0 equals the ridge map
        k0, _ = ridge(extract(source, seq))
        k1, _ = proj(extract(source, seq))
    torch.testing.assert_close(k0[0], k1[0], atol=1e-5, rtol=1e-5)
    data = batches([seq[0].tolist()] * 4, batch=2, device=torch.device("cpu"))
    losses = train(_toy_cfg(tmp_path), source, tgt, proj, data, log=lambda *_: None)
    assert len(losses) == 6 and all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0], losses
    assert (tmp_path / "run" / "last.pt").exists() and (tmp_path / "run" / "step_6" / "mlp.safetensors").exists()


def test_checkpoint_save_resume(source, target, tmp_path):
    ridge = fit_ridge(source, target, _fit_batches(torch.randint(0, VOCAB, (1, 48)), n=4))
    proj = MLPProjector.from_ridge(ridge, bottleneck=16)
    seqs = [torch.randint(0, VOCAB, (48,)).tolist() for _ in range(4)]
    cfg = _toy_cfg(tmp_path, steps=3, save_every=3)
    train(cfg, source, target, proj, batches(seqs, 2, torch.device("cpu")), log=lambda *_: None)
    # resume: new projector object, continue to step 5
    proj2 = MLPProjector.from_ridge(ridge, bottleneck=16)
    opt = make_optimizer(proj2, cfg)
    assert load_checkpoint(tmp_path / "run" / "last.pt", proj2, opt) == 3
    for a, b in zip(proj.parameters(), proj2.parameters()):
        torch.testing.assert_close(a, b)
    cfg2 = _toy_cfg(tmp_path, steps=5, save_every=5, resume=True)
    losses = train(cfg2, source, target, proj2, batches(seqs, 2, torch.device("cpu")), log=lambda *_: None)
    assert len(losses) == 2  # only steps 4 and 5 ran
    back = MLPProjector.load(tmp_path / "run" / "step_5")
    for a, b in zip(back.parameters(), proj2.parameters()):
        torch.testing.assert_close(a, b)


def test_eval_arms_are_consistent(target):
    ids = torch.randint(0, VOCAB, (2, 20))
    with torch.no_grad():
        none = continuation_loss(target, ids[:, 10:])
        orc = continuation_loss(target, ids[:, 10:], oracle_cache(target, ids[:, :10]), 10)
        per_tok = continuation_loss(target, ids[:, 10:], None, 0, reduce=False)
    assert per_tok.shape == (2, 9) and torch.isfinite(none) and torch.isfinite(orc)
    torch.testing.assert_close(per_tok.mean(), none)
