"""Different-context QA (Latent Cache Flow's LCF-X setting) from HotpotQA distractor validation:
the two supporting paragraphs are split, the sender reads one (`context`), the receiver reads the other
(`context_receiver`) plus the question. Only the union answers the question.

  uv run python -m k2cascade.projector.data.prepare_hotpot --n 600 --out data/hotpot_split.jsonl
"""
from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/hotpot_split.jsonl")
    a = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").shuffle(seed=a.seed)
    rng = random.Random(a.seed)
    n = 0
    with open(a.out, "w") as f:
        for r in ds:
            if r["type"] != "bridge":
                continue
            titles = r["context"]["title"]; sents = r["context"]["sentences"]
            sup = list(dict.fromkeys(r["supporting_facts"]["title"]))
            if len(sup) != 2 or any(t not in titles for t in sup):
                continue
            paras = {t: " ".join(sents[titles.index(t)]) for t in sup}
            order = sup[:] if rng.random() < 0.5 else sup[::-1]
            f.write(json.dumps({"id": r["id"], "question": r["question"], "answers": [r["answer"]],
                                "context": paras[order[0]], "context_receiver": paras[order[1]],
                                "sender_title": order[0], "receiver_title": order[1]}) + "\n")
            n += 1
            if n >= a.n:
                break
    print(f"wrote {n} bridge questions to {a.out}")


if __name__ == "__main__":
    import os, sys
    main(); sys.stdout.flush(); os._exit(0)
