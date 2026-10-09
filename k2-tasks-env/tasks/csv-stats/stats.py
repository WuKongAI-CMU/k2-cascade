"""Tiny CSV summary. Usage: python stats.py FILE COLUMN"""
import sys


def summarize(path, column):
    with open(path) as f:
        lines = f.read().splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    return {"rows": len(rows)}


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 1
    s = summarize(argv[0], argv[1])
    for k, v in s.items():
        print(f"{k} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
