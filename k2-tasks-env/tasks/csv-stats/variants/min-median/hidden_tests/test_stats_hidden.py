import stats


def write(tmp_path, text):
    p = tmp_path / "d.csv"; p.write_text(text); return str(p)


def test_median_even_count_is_mean_of_middle_two(tmp_path):
    s = stats.summarize(write(tmp_path, "n,v\na,1\nb,4\nc,2\nd,3\n"), "v")
    assert s["median"] == 2.5 and s["min"] == 1.0


def test_other_column(tmp_path):
    s = stats.summarize(write(tmp_path, "id,price,qty\n1,9.5,2\n2,0.5,10\n3,3,7\n"), "price")
    assert s["min"] == 0.5 and s["median"] == 3.0


def test_negative_and_unsorted(tmp_path):
    s = stats.summarize(write(tmp_path, "n,v\na,5\nb,-4\nc,2\n"), "v")
    assert s["min"] == -4.0 and s["median"] == 2.0


def test_blank_lines_everywhere(tmp_path):
    s = stats.summarize(write(tmp_path, "n,v\n\na,4\n\n\nb,6\n\n"), "v")
    assert s["rows"] == 2 and s["median"] == 5.0
