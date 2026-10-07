import stats


def write(tmp_path, text):
    p = tmp_path / "data.csv"
    p.write_text(text)
    return str(p)


def test_counts_rows(tmp_path):
    p = write(tmp_path, "name,score\na,10\nb,20\nc,30\n")
    assert stats.summarize(p, "score")["rows"] == 3


def test_main_prints_row_count(tmp_path, capsys):
    p = write(tmp_path, "name,score\na,10\nb,20\n")
    assert stats.main([p, "score"]) == 0
    assert "rows 2" in capsys.readouterr().out


def test_mean_and_max_of_column(tmp_path):
    p = write(tmp_path, "name,score\na,10\nb,20\nc,30\n")
    s = stats.summarize(p, "score")
    assert s["mean"] == 20.0
    assert s["max"] == 30.0


def test_skips_blank_lines(tmp_path):
    p = write(tmp_path, "name,score\na,10\n\nb,20\n\nc,30\n")
    s = stats.summarize(p, "score")
    assert s["rows"] == 3
    assert s["mean"] == 20.0
