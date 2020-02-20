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

Install with ``pip`` from Github::

    pip install https://github.com/simonsobs/mapsims/archive/master.zip

Development installation
========================

Development install::

    git clone https://github.com/simonsobs/mapsims
    cd mapsims
    pip install -e .

Run unit tests::

    python setup.py test -V

Run Jupyter notebook tests (requires ``nbval``)::

    bash run_notebook_tests.sh

Build documentation::

    python setup.py build_docs
