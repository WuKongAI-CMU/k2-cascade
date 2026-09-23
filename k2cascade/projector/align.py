"""Cross-tokenizer sender: when the sender and receiver do not share a vocabulary, the receiver's token
positions are aligned to the sender's by character offsets. Receiver position t is served by the first sender
token whose character span ends at or after the end of receiver token t (the sender has "read that far").

Attach with `attach_aligner(src_model, src_tokenizer, tgt_tokenizer)`; `extract()` then transparently decodes
the receiver ids, re-tokenises for the sender, runs the sender, and gathers positions.
"""
from __future__ import annotations

import bisect

import torch


def align_indices(src_offsets: list[tuple[int, int]], tgt_offsets: list[tuple[int, int]]) -> list[int]:
    """For each receiver token, the index of the sender token whose end >= the receiver token's end."""
    ends = [e for _, e in src_offsets]
    out = []
    for _, e in tgt_offsets:
        j = bisect.bisect_left(ends, e)
        out.append(min(j, len(ends) - 1))
    return out


class Aligner:
    def __init__(self, src_tok, tgt_tok):
        self.src_tok, self.tgt_tok = src_tok, tgt_tok
        self.src_bos = [src_tok.bos_token_id] if getattr(src_tok, "bos_token_id", None) is not None else []

    def plan(self, tgt_ids: list[int]) -> tuple[list[int], list[int]]:
        """Sender ids and, per receiver position, the sender position to read."""
        special = {t for t in (self.tgt_tok.bos_token_id, self.tgt_tok.eos_token_id, self.tgt_tok.pad_token_id) if t is not None}
        keep = [i for i, t in enumerate(tgt_ids) if t not in special]
        text = self.tgt_tok.decode([tgt_ids[i] for i in keep])
        # receiver offsets on the decoded text: re-tokenise it (round trip is exact for these tokenizers up to
        # the special tokens we removed); fall back to a length-proportional map if the lengths disagree
        t_enc = self.tgt_tok(text, add_special_tokens=False, return_offsets_mapping=True)
        s_enc = self.src_tok(text, add_special_tokens=False, return_offsets_mapping=True)
        src_ids = self.src_bos + s_enc["input_ids"]
        off = len(self.src_bos)
        if len(t_enc["input_ids"]) == len(keep):
            idx_kept = [off + j for j in align_indices(s_enc["offset_mapping"], t_enc["offset_mapping"])]
        else:
            n_s, n_t = len(s_enc["input_ids"]), len(keep)
            idx_kept = [off + min(n_s - 1, (i * n_s) // max(n_t, 1)) for i in range(n_t)]
        idx = [0] * len(tgt_ids)
        for pos, j in zip(keep, idx_kept):
            idx[pos] = j
        # special receiver tokens (BOS) read the sender's first position
        return src_ids, idx


def attach_aligner(src_model, src_tok, tgt_tok) -> None:
    src_model._k2_aligner = Aligner(src_tok, tgt_tok)


def gather_bundle(bundle, idx: torch.Tensor):
    """Select sender positions `idx` (B, T_tgt) from every layer's (B, H, T_src, D)."""
    from .extract import KVBundle
    def g(x):
        B, H, _, D = x.shape
        ii = idx[:, None, :, None].expand(B, H, idx.shape[1], D).to(x.device)
        return x.gather(2, ii)
    return KVBundle(keys=[g(k) for k in bundle.keys], values=[g(v) for v in bundle.values],
                    hidden=[h.gather(1, idx[:, :, None].expand(-1, -1, h.shape[-1]).to(h.device)) for h in bundle.hidden])


def extract_aligned(model, tgt_ids: torch.Tensor, with_hidden: bool):
    """extract() for a sender with an attached Aligner: run per row, gather to receiver positions, stack."""
    from .extract import extract, KVBundle
    al = model._k2_aligner
    outs = []
    for row in tgt_ids.tolist():
        src_ids, idx = al.plan(row)
        b = extract.__wrapped__(model, torch.tensor([src_ids], device=tgt_ids.device), None, with_hidden)
        outs.append(gather_bundle(b, torch.tensor([idx], device=tgt_ids.device)))
    return KVBundle(keys=[torch.cat([o.keys[j] for o in outs]) for j in range(outs[0].num_layers)],
                    values=[torch.cat([o.values[j] for o in outs]) for j in range(outs[0].num_layers)],
                    hidden=[torch.cat([o.hidden[j] for o in outs]) for j in range(len(outs[0].hidden))])
