import torch

from k2cascade.projector.align import Aligner, align_indices, attach_aligner
from k2cascade.projector.extract import extract


def test_align_indices_reads_up_to_receiver_token_end():
    src = [(0, 3), (3, 6), (6, 10)]          # "abc" "def" "ghij"
    tgt = [(0, 2), (2, 4), (4, 10)]          # "ab" "cd" "efghij"
    assert align_indices(src, tgt) == [0, 1, 2]


class CharTok:
    """Toy fast-tokenizer look-alike: fixed-width character chunks with offsets."""
    bos_token_id, eos_token_id, pad_token_id = 1, None, None

    def __init__(self, width):
        self.w = width

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offs = [], []
        for i in range(0, len(text), self.w):
            ids.append(2 + (sum(map(ord, text[i:i + self.w])) % 20)); offs.append((i, min(i + self.w, len(text))))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def decode(self, ids):
        return "x" * (len(ids) * self.w)


def test_aligned_extract_matches_receiver_length(source):
    src_tok, tgt_tok = CharTok(3), CharTok(2)
    attach_aligner(source, src_tok, tgt_tok)
    try:
        ids = torch.tensor([[1, 5, 6, 7, 8, 9]])  # BOS + 5 receiver tokens = 10 chars -> 4 sender tokens
        b = extract(source, ids, with_hidden=True)
        assert b.keys[0].shape[2] == 6 and b.hidden[0].shape[1] == 6
        plan_ids, idx = source._k2_aligner.plan(ids[0].tolist())
        assert len(plan_ids) == 1 + 4 and idx == [0, 1, 2, 2, 3, 4]
    finally:
        source._k2_aligner = None
