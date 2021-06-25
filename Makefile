test: test_basic

test_basic:
	python -m pytest -v -ra -m "not slow"

test_full:
	python -m pytest -v -ra