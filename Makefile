test: test_basic

test_basic:
	python3 -m pytest -v -ra -m "not slow"

test_full:
	python3 -m pytest -v -ra

test_debug:
	python3 -m pytest -v -ra --capture=no