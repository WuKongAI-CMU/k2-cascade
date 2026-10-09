import jlens
import torch

from k2cascade.projector.jlens_fit import K2LensModel


def test_fit_on_toy(target):
    model = K2LensModel(target)
    prompt = " ".join(str(i % 50 + 1) for i in range(24))
    lens = jlens.fit(model, [prompt, prompt], dim_batch=8, max_seq_len=24)
    J = lens.jacobians[0]
    assert J.shape == (model.d_model, model.d_model) and torch.isfinite(J).all() and J.abs().sum() > 0
    h = torch.randn(3, model.d_model)
    assert model.unembed(lens.transport(h, 0)).shape[0] == 3
