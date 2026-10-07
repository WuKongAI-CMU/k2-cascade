import wc


def write(tmp_path, text):
    p = tmp_path / "in.txt"; p.write_text(text); return str(p)


def test_lines_and_words(tmp_path, capsys):
    assert wc.main([write(tmp_path, "one two\nthree four five\n")]) == 0
    out = capsys.readouterr().out
    assert "lines 2" in out and "words 5" in out


def test_words_only(tmp_path, capsys):
    assert wc.main(["--words", write(tmp_path, "one two\nthree four five\n")]) == 0
    assert capsys.readouterr().out == "words 5\n"


def test_longest_flag(tmp_path, capsys):
    assert wc.main(["--longest", write(tmp_path, "one three\nfour\n")]) == 0
    assert capsys.readouterr().out == "longest three\n"


def test_bytes_flag(tmp_path, capsys):
    assert wc.main(["--bytes", write(tmp_path, "ab\n")]) == 0
    assert capsys.readouterr().out == "bytes 3\n"
