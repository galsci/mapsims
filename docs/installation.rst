*****************************
Requirements and installation
*****************************


Requirements
============

* PySM 3 from https://github.com/healpy/pysm
* ``numba``
* ``h5py`` to read custom instrument parameter files
* ``healpy``
* ``so_pysm_models``
* ``so_noise_models``
* ``toml``

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

Build documentation::

    python setup.py build_docs
