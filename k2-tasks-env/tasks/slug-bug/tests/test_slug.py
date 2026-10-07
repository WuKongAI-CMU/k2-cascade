from slug import slugify


def test_lowercases():
    assert slugify("Hello") == "hello"


def test_spaces_become_dashes():
    assert slugify("hello world") == "hello-world"


def test_collapses_repeated_dashes():
    assert slugify("hello   world") == "hello-world"
    assert slugify("a -- b") == "a-b"


def test_strips_leading_and_trailing_dashes():
    assert slugify("  Hello, World!  ") == "hello-world"
    assert slugify("--x--") == "x"
