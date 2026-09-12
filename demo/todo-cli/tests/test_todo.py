import json
import todo


def setup_function(_):
    if todo.DB.exists():
        todo.DB.unlink()


def test_add_and_list():
    todo.add("buy milk")
    todo.add("write code")
    assert [i["text"] for i in todo.list_items()] == ["buy milk", "write code"]


def test_done_marks_item():
    todo.add("a")
    todo.done(1)
    assert todo.list_items()[0]["done"] is True


def test_list_pending_flag_hides_done_items():
    todo.add("a")
    todo.add("b")
    todo.done(1)
    assert [i["text"] for i in todo.list_items(["--pending"])] == ["b"]


def test_list_done_flag_shows_only_done_items():
    todo.add("a")
    todo.add("b")
    todo.done(2)
    assert [i["text"] for i in todo.list_items(["--done"])] == ["b"]
