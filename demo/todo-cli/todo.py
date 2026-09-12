"""Tiny todo CLI. Usage: python todo.py add "text" | list | done N"""
import json
import sys
from pathlib import Path

DB = Path(__file__).with_name("todo.json")


def load():
    return json.loads(DB.read_text()) if DB.exists() else []


def save(items):
    DB.write_text(json.dumps(items, indent=2))


def add(text):
    items = load()
    items.append({"id": len(items) + 1, "text": text, "done": False})
    save(items)
    return items[-1]


def done(item_id):
    items = load()
    for it in items:
        if it["id"] == item_id:
            it["done"] = True
    save(items)


def list_items(args=None):
    return load()


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    cmd, rest = argv[0], argv[1:]
    if cmd == "add":
        it = add(" ".join(rest))
        print(f"added #{it['id']}")
    elif cmd == "done":
        done(int(rest[0]))
        print("ok")
    elif cmd == "list":
        for it in list_items(rest):
            mark = "x" if it["done"] else " "
            print(f"[{mark}] {it['id']} {it['text']}")
    else:
        print(f"unknown command {cmd}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
