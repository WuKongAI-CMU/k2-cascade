from k2cascade.projector.variants import build, build_variants, gold_sentence_index, split_sentences

CTX = "The library opened in 1911. It was extended twice since then. A glass wing faces the park."


def test_split_and_gold_index():
    s = split_sentences(CTX)
    assert len(s) == 3
    assert gold_sentence_index(s, CTX.index("1911")) == 0
    assert gold_sentence_index(s, CTX.index("park")) == 2


def test_build_variants_inserts_after_gold_and_removes_gold():
    v = build_variants(CTX, CTX.index("1911"), {"counter_answer": "1923", "sentence": "The library opened in 1923."})
    assert v["contradicted"].startswith("The library opened in 1911. The library opened in 1923. It was")
    assert v["removed"] == "It was extended twice since then. A glass wing faces the park."
    assert v["gold_sentence"] == "The library opened in 1911."


def test_build_drops_failed_judgements():
    ex = [{"context": CTX, "question": "When did the library open?", "answers": ["1911"], "answer_start": CTX.index("1911")},
          {"context": CTX, "question": "What faces the park?", "answers": ["A glass wing"], "answer_start": CTX.index("A glass")}]
    calls = []
    def judge(c, q, a):
        calls.append(q)
        return None if "park" in q else {"counter_answer": "1923", "sentence": "The library opened in 1923."}
    out = build(ex, judge, workers=2)
    assert len(calls) == 2 and len(out) == 1 and out[0]["counter_answer"] == "1923" and "removed" in out[0]
