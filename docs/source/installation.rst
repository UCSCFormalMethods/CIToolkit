Installation
============

``citoolkit`` is writtin in Python 3, and requires **Python 3.10**. It is available as a PyPi package and can be installed by running:

.. code-block:: console

    $ pip install citoolkit


If you want to modify ``citoolkit``, access the test suite, or run the included examples/experiments you can use [Poetry](https://python-poetry.org/ "Poetry"). First install Poetry and optionally activate the virtual environment in which you would like to run ``citoolkit``. Then navigate to the ``citoolkit`` root directory and run:

.. code-block:: console

    $ poetry install


Alternatively, to automatically install ``citoolkit`` and all its dependencies in a virtual environment, navigate to the ``citoolkit`` root directory and run:

.. code-block:: console

    $ make create_env

.. note::

	At the moment the packages ``pyapproxmc`` and ``pyunigen`` are not available via PyPi, so they must be installed into your Python environment manually. They are available on GitHub: `ApproxMC <https://github.com/meelgroup/approxmc>`_ `Unigen <https://github.com/meelgroup/unigen>`_.

You can then activate that virtual environment by running:

.. code-block:: console

    $ make enter_env
