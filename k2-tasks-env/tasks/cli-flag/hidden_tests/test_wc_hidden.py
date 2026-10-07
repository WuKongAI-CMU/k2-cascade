import wc


def write(tmp_path, text):
    p = tmp_path / "in.txt"; p.write_text(text); return str(p)


def test_chars_counts_newlines(tmp_path, capsys):
    assert wc.main(["--chars", write(tmp_path, "ab\n")]) == 0
    assert capsys.readouterr().out == "chars 3\n"


def test_top_more_than_available(tmp_path, capsys):
    assert wc.main(["--top", "5", write(tmp_path, "x y x\n")]) == 0
    assert capsys.readouterr().out == "x 2\ny 1\n"


def test_top_ties_are_stable_by_first_occurrence(tmp_path, capsys):
    assert wc.main(["--top", "2", write(tmp_path, "b a b a\n")]) == 0
    assert capsys.readouterr().out == "b 2\na 2\n"


def test_existing_flags_unchanged(tmp_path, capsys):
    assert wc.main(["--lines", write(tmp_path, "a\nb\n")]) == 0
    assert capsys.readouterr().out == "lines 2\n"
