Demo task (copy into --task):

The test suite in this repo fails. Run `python -m pytest -q` to see the failures, then extend
`summarize` in stats.py so that it also returns the mean and max of the given numeric column
and skips blank lines, so that all tests pass. Do not modify the tests.
