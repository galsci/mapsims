*****************************
Requirements and installation
*****************************


Requirements
============

* PySM 3 from https://github.com/healpy/pysm (install with `conda install -c conda-forge pysm3`)
* ``numba``
* ``h5py`` to read instruments parameters other than SO
* ``healpy``
* ``so_pysm_models``
* ``so_noise_models``
* ``toml``
* ``pytest`` to run unit tests
* ``nbval`` to run Notebook unit tests

Installation
============

Install the last packaged release with ``pip`` from PyPI::

    pip install mapsims

Install the development version with ``pip`` from Github::

    pip install https://github.com/simonsobs/mapsims/archive/master.zip

Development installation
========================

Unfortunately, a "editable" installation is not supported anymore.
I recommend to run tests from the repository base folder.

Or if you are testing in a notebook, add the package root to `sys.path`.

Clone the repository::

    git clone https://github.com/simonsobs/mapsims
    cd mapsims

Make sure you have the requirements installed, easiest is
to install the last version of the package from PyPI
and then uninstall it::

    pip install mapsims
    pip uninstall mapsims

Make edits and run unit tests::

    pytest -v

Run Jupyter notebook tests (requires ``nbval``)::

    bash run_notebook_tests.sh

Build documentation::

    cd docs
    make html

Manage dependencies
===================

In case you need to modify dependencies, you will need to `install poetry <https://python-poetry.org/docs/#installation>`_

In case you are using an anaconda environment, `poetry` will try to use that to install the requirements of `mapsims`,
instead if you want to keep it isolated, better create a dedicated virtualenv::

    python -m venv venv
    poetry use venv/bin/python

Setup the dedicated virtual environment for the project and include the requirements for testing::

    poetry install --extras test

Then run the tests within that environment::

    poetry run pytest
