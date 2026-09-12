"""Tiny wc. Usage: python wc.py [--lines] [--words] FILE"""
import argparse
import sys


def count(text):
    return {"lines": len(text.splitlines()), "words": len(text.split())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--lines", action="store_true")
    ap.add_argument("--words", action="store_true")
    a = ap.parse_args(argv)
    with open(a.file) as f:
        text = f.read()
    c = count(text)
    selected = [k for k in ("lines", "words") if getattr(a, k)] or ["lines", "words"]
    for k in selected:
        print(f"{k} {c[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
