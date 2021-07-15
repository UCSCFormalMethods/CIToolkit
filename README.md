# Control Improvisation Toolkit

## Introduction
`citoolkit` is a library containing tools to create and solve instances of the Control Improvisation problem and its extensions. This library supports the following flavors of Control Improvisation:

- Control Improvisation (CI) [[Fremont et al. 2017]](https://arxiv.org/abs/1704.06319 "[Fremont et al, 2017]")

The following specification types are currently supported for hard and soft constraints:

- Deterministic Finite Automaton (DFA)

If you encounter any problems with the `citoolkit` please submit an issue on GitHub or contact Eric at <evin@ucsc.edu>

## Setup
### Installation
``citoolkit`` is writtin in Python 3, and requires **Python 3.9**. It is available as a PyPi package and can be installed by running:

```shell
pip install citoolkit
```

If you want to modify `citoolkit`, access the test suite, or run the included examples/experiments you can use [Poetry](https://python-poetry.org/ "Poetry"). First install Poetry and optionally activate the virtual environment in which you would like to run `citoolkit`. Then navigate to the `citoolkit` root directory and run:

```shell
poetry install
```

Alternatively, simply type `make` in the `citoolkit` root directory to automatically install `citoolkit` and all its dependencies in a virtual environment, and then activate that virtual environment.

### Tests
To run the basic test suite, navigate to the `citoolkit` root directory and run:

```shell
make test
```

To run the full test suite, navigate to the `citoolkit` root directory and run:

```shell
make test_full
```

**Note:** The basic test suite is designed to run quickly, while the full test suite tests more extensively. As a result, the full test suite can take *significantly* longer to run than the basic test suite.

## Documentation
Documentation for `citoolkit` can be found on [Read the Docs](https://citoolkit.readthedocs.io/en/latest/)
