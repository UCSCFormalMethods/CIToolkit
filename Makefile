RUN_PYTEST = poetry run pytest -v -ra

.PHONY: enter_env make_env update test test_basic test_full test_basic_debug test_full_debug docs build clean

enter_env: make_env
	poetry shell

make_env:
	poetry install -E dev

update:
	poetry update

test: test_basic

test_basic: make_env
	$(RUN_PYTEST) -m "not slow"

test_full: make_env
	$(RUN_PYTEST)

test_basic_debug: make_env
	$(RUN_PYTEST) --capture=no -m "not slow"

test_full_debug: make_env
	$(RUN_PYTEST) --capture=no

docs: make_env
	poetry run make -C docs clean
	poetry run make -C docs api_doc
	poetry run make -C docs html

clean:
	rm -rf docs/build/*
	rm -rf dist

build: clean test_full docs
	poetry build

publish: build
	poetry publish
