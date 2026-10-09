import stats


def write(tmp_path, text):
    p = tmp_path / "data.csv"; p.write_text(text); return str(p)


def test_counts_rows(tmp_path):
    assert stats.summarize(write(tmp_path, "name,score\na,10\nb,20\nc,30\n"), "score")["rows"] == 3


def test_main_prints_row_count(tmp_path, capsys):
    assert stats.main([write(tmp_path, "name,score\na,10\nb,20\n"), "score"]) == 0
    assert "rows 2" in capsys.readouterr().out


def test_min_and_median_of_column(tmp_path):
    s = stats.summarize(write(tmp_path, "name,score\na,10\nb,30\nc,20\n"), "score")
    assert s["min"] == 10.0 and s["median"] == 20.0


def test_skips_blank_lines(tmp_path):
    s = stats.summarize(write(tmp_path, "name,score\na,10\n\nb,20\n\nc,30\n"), "score")
    assert s["rows"] == 3 and s["median"] == 20.0
