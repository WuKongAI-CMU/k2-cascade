from slug import slugify


def test_digits_kept():
    assert slugify("Top 10 Tips!") == "top_10_tips"


def test_dashes_become_underscores():
    assert slugify("a--b..c") == "a_b_c"


def test_all_punctuation_is_empty():
    assert slugify("!!!") == ""


def test_unicode_letters_dropped():
    assert slugify("café au lait") == "caf_au_lait"
