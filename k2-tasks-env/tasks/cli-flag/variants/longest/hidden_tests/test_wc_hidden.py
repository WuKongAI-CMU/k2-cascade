import wc


def write(tmp_path, text):
    p = tmp_path / "in.txt"; p.write_text(text); return str(p)


def test_longest_tie_is_first_occurrence(tmp_path, capsys):
    assert wc.main(["--longest", write(tmp_path, "abc def\n")]) == 0
    assert capsys.readouterr().out == "longest abc\n"


def test_bytes_counts_utf8_bytes_not_chars(tmp_path, capsys):
    p = tmp_path / "in.txt"; p.write_text("é\n", encoding="utf-8")
    assert wc.main(["--bytes", str(p)]) == 0
    assert capsys.readouterr().out == "bytes 3\n"


def test_longest_on_empty_file_prints_empty(tmp_path, capsys):
    assert wc.main(["--longest", write(tmp_path, "")]) == 0
    assert capsys.readouterr().out == "longest \n"


def test_existing_flags_unchanged(tmp_path, capsys):
    assert wc.main(["--lines", write(tmp_path, "a\nb\n")]) == 0
    assert capsys.readouterr().out == "lines 2\n"
