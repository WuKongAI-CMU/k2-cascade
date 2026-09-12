"""URL slugs. Usage: python slug.py "Some Title" """
import re
import sys


def slugify(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9]", "-", text)


if __name__ == "__main__":
    print(slugify(" ".join(sys.argv[1:])))
