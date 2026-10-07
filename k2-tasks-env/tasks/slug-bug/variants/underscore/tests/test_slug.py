from slug import slugify


def test_lowercases():
    assert slugify("Hello") == "hello"


def test_spaces_become_underscores():
    assert slugify("hello world") == "hello_world"


def test_collapses_repeated_separators():
    assert slugify("hello   world") == "hello_world"
    assert slugify("a -- b") == "a_b"


def test_strips_leading_and_trailing():
    assert slugify("  Hello, World!  ") == "hello_world"
    assert slugify("__x__") == "x"
