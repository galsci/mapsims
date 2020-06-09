*********************
mapsims Documentation
*********************

``mapsims`` is a Python 3 package to produce map based simulations for the `Simons Observatory <https://simonsobservatory.org/>`_
or other CMB experiments.

It creates simulated maps in HEALPix and CAR pixelization based on:

* Foreground models included in PySM
* Custom foregrounds models from the :py:mod:`so_pysm_models` package
* Precomputed Cosmic Microwave Background simulations
* Noise simulations based on expected performance and simulated hitmaps
* Effect of gaussian beam convolution

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   models

Reference/API
=============

.. automodapi:: mapsims
