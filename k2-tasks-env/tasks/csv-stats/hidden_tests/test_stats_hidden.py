import stats


def write(tmp_path, text):
    p = tmp_path / "d.csv"; p.write_text(text); return str(p)


def test_mean_is_float_not_truncated(tmp_path):
    s = stats.summarize(write(tmp_path, "n,v\na,1\nb,2\n"), "v")
    assert s["mean"] == 1.5 and s["max"] == 2.0


def test_other_column_name(tmp_path):
    s = stats.summarize(write(tmp_path, "id,price,qty\n1,9.5,2\n2,0.5,10\n"), "price")
    assert s["rows"] == 2 and s["mean"] == 5.0 and s["max"] == 9.5


def test_blank_lines_at_end_and_middle(tmp_path):
    s = stats.summarize(write(tmp_path, "n,v\n\na,4\n\n\nb,6\n\n"), "v")
    assert s["rows"] == 2 and s["mean"] == 5.0


def test_negative_values(tmp_path):
    s = stats.summarize(write(tmp_path, "n,v\na,-4\nb,2\n"), "v")
    assert s["max"] == 2.0 and s["mean"] == -1.0
