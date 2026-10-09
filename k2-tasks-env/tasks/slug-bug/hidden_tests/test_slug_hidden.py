from slug import slugify


def test_digits_kept():
    assert slugify("Top 10 Tips!") == "top-10-tips"


def test_underscores_and_punctuation_collapse():
    assert slugify("a__b..c") == "a-b-c"


def test_all_punctuation_is_empty():
    assert slugify("!!!") == ""


def test_unicode_letters_dropped():
    assert slugify("café au lait") == "caf-au-lait"
