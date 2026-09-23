"""Keep contradicted variants whose inserted sentence entails the counter-answer and does not entail the gold
(Xie et al. 2023 / ConflictBank filter). Writes the surviving rows plus an `nli` field.

uv run python -m k2cascade.projector.nli_filter --data data/squad_variants.jsonl --out data/squad_variants_nli.jsonl
"""
from __future__ import annotations

import json


def main(argv=None) -> None:
    import argparse
    import torch
    from .sender_entropy import NLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--nli", default="microsoft/deberta-large-mnli")
    a = ap.parse_args(argv)
    nli = NLI(a.nli, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    kept = total = 0
    with open(a.out, "w") as f:
        for line in open(a.data):
            ex = json.loads(line); total += 1
            hyp_c = f"The answer to '{ex['question']}' is {ex['counter_answer']}."
            hyp_g = f"The answer to '{ex['question']}' is {ex['answers'][0]}."
            ok = nli.entails(ex["sentence"], hyp_c) and not nli.entails(ex["sentence"], hyp_g)
            ex["nli"] = ok
            if ok:
                kept += 1; f.write(json.dumps(ex) + "\n")
    print(f"kept {kept}/{total}")


if __name__ == "__main__":
    main()
