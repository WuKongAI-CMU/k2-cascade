"""Stream HuggingFaceFW/fineweb-edu (sample-10BT) and write N packed sequences of `seq_len` tokens.

  uv run --extra data python -m k2cascade.projector.data.prepare_fineweb --n 2048 --out data/fineweb_1024.jsonl

Each JSONL line: {"ids": [...seq_len ints], "text": decoded text}. Tokenizer: the 7B (same vocab as 3.7B).
Documents are concatenated with EOS and cut into fixed windows (pretraining-style packing).
Do not run on the laptop: it downloads shards.
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--out", default="data/fineweb_1024.jsonl")
    ap.add_argument("--tokenizer", default="IFM/K2-Horizon-7B")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 1
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
    buf, written = [], 0
    with open(args.out, "w") as f:
        for row in ds:
            buf.extend(tok(row["text"], add_special_tokens=False)["input_ids"] + [eos])
            while len(buf) >= args.seq_len and written < args.n:
                ids, buf = buf[:args.seq_len], buf[args.seq_len:]
                f.write(json.dumps({"ids": ids, "text": tok.decode(ids)}) + "\n")
                written += 1
            if written >= args.n:
                break
    print(f"wrote {written} sequences of {args.seq_len} tokens to {args.out}")


if __name__ == "__main__":
    main()
