import todo


def setup_function(_):
    if todo.DB.exists():
        todo.DB.unlink()


def test_pending_with_nothing_done_returns_all():
    todo.add("a"); todo.add("b")
    assert [i["text"] for i in todo.list_items(["--open"])] == ["a", "b"]


def test_done_with_nothing_done_returns_empty():
    todo.add("a")
    assert todo.list_items(["--closed"]) == []


def test_filters_keep_ids():
    todo.add("a"); todo.add("b"); todo.add("c")
    todo.done(2)
    assert [i["id"] for i in todo.list_items(["--open"])] == [1, 3]
    assert [i["id"] for i in todo.list_items(["--closed"])] == [2]


def test_no_args_still_lists_everything():
    todo.add("a"); todo.done(1); todo.add("b")
    assert len(todo.list_items()) == 2
