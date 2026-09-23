"""Wall-clock cost of the handoff: receiver re-prefill of the prefix vs sender prefill + projector forward.

uv run python -m k2cascade.projector.latency --projector runs/mlp/step_1500 --prefix 512 --reps 20
"""
from __future__ import annotations

import json
import time

import torch


@torch.no_grad()
def timeit(fn, reps: int, dev) -> float:
    for _ in range(3):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main(argv=None) -> None:
    import argparse
    from pathlib import Path
    from .extract import extract
    from .mlp import MLPProjector
    from .ridge import RidgeProjector
    from .train import TrainConfig, freeze, load_models
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="IFM/K2-Horizon-3.7B"); ap.add_argument("--target", default="IFM/K2-Horizon-7B")
    ap.add_argument("--projector", required=True); ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--reps", type=int, default=20); ap.add_argument("--out", default="analysis/latency.json")
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, tgt, tok = load_models(TrainConfig(source=a.source, target=a.target), dev)
    freeze(src), freeze(tgt)
    proj = (MLPProjector.load if (Path(a.projector) / "mlp.safetensors").exists() else RidgeProjector.load)(a.projector, dev)
    ids = torch.randint(5, 1000, (1, a.prefix), device=dev)
    res = {"prefix": a.prefix, "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"}
    res["receiver_prefill_s"] = timeit(lambda: tgt(input_ids=ids, use_cache=True), a.reps, dev)
    res["sender_prefill_s"] = timeit(lambda: extract(src, ids, with_hidden=False), a.reps, dev)
    b = extract(src, ids, with_hidden=False)
    res["projector_forward_s"] = timeit(lambda: proj(b), a.reps, dev)
    res["handoff_s"] = res["projector_forward_s"]  # what the receiver waits for, given the sender already read
    res["speedup_vs_reprefill"] = res["receiver_prefill_s"] / res["handoff_s"]
    res["projector_params_M"] = sum(p.numel() for p in proj.parameters()) / 1e6 if hasattr(proj, "parameters") else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
