import wc


def write(tmp_path, text):
    p = tmp_path / "in.txt"
    p.write_text(text)
    return str(p)


def test_lines_and_words(tmp_path, capsys):
    p = write(tmp_path, "one two\nthree four five\n")
    assert wc.main([p]) == 0
    out = capsys.readouterr().out
    assert "lines 2" in out
    assert "words 5" in out


def test_words_only(tmp_path, capsys):
    p = write(tmp_path, "one two\nthree four five\n")
    assert wc.main(["--words", p]) == 0
    assert capsys.readouterr().out == "words 5\n"


def test_chars_flag(tmp_path, capsys):
    p = write(tmp_path, "one two\nthree\n")
    assert wc.main(["--chars", p]) == 0
    assert capsys.readouterr().out == "chars 14\n"


def test_top_words(tmp_path, capsys):
    p = write(tmp_path, "a b a c a b\n")
    assert wc.main(["--top", "2", p]) == 0
    assert capsys.readouterr().out == "a 3\nb 2\n"
