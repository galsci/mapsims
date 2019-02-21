*********************
mapsims Documentation
*********************

``mapsims`` is a Python 3 package to produce map based simulations for the `Simons Observatory <https://simonsobservatory.org/>`_.

It creates simulated Simons Observatory maps based on:

* Foreground models included in PySM
* Custom foregrounds models from the ``so_pysm_models`` package, currently ``GaussianDust`` and ``GaussianSynchrotron``
* Precomputed Cosmic Microwave Background simulations
* Noise simulations based on expected performance and simulated hitmaps with either classical or opportunistic scanning strategy
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
