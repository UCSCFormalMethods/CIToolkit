RUN_PYTEST = poetry run pytest -l -v -ra --randomly-dont-reorganize --log-level=DEBUG

.PHONY: enter_env make_env update check test test_basic test_full format docs build clean publish

enter_env: make_env
	poetry shell

make_env:
	poetry install -E dev

update:
	poetry update

check: format test_full

test: test_basic

test_basic: make_env
	$(RUN_PYTEST) -m "not advanced"

test_full: make_env
	$(RUN_PYTEST) --durations=5 --cov-report term:skip-covered --cov=citoolkit

format:
	black --preview citoolkit/

docs: make_env
	poetry run make -C docs clean
	poetry run make -C docs api_doc
	poetry run make -C docs html

clean:
	rm -rf docs/build/*
	rm -rf dist

build: clean docs
	poetry build

publish: build
	poetry publish
