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

Requirements
============

TODO

Installation
============

TODO


Usage
=====


``mapsims`` can be used with a configuration file or using directly Python classes.

See the ``example_config.cfg`` file included in the package, create the simulator object with::

    import mapsims
    simulator = mapsims.from_config("example_config.cfg")

produce the output map with:

    output_map = simulator.execute()

Using instead the Python classes, we first need to create the custom component objects, as
an example we will use all defaults options::

    NSIDE = 16
    dust = so_pysm_models.GaussianDust(
        target_nside=NSIDE,
    )

    sync = so_pysm_models.GaussianSynchrotron(
        target_nside=NSIDE,
    )

Then we can create a ``SONoiseSimulator``, the most important parameter is the scanning strategy,
it can be either "classical" or "opportunistic"::

    noise = mapsims.SONoiseSimulator(
        telescope="SA",
        band=27,
        nside=NSIDE,
        return_uK_CMB=True,
        sensitivity_mode="baseline",
        apply_beam_correction=True,
        apply_kludge_correction=True,
        scanning_strategy="classical",
        LA_number_LF=1,
        LA_number_MF=4,
        LA_number_UHF=2,
        SA_years_LF=1,
        SA_one_over_f_mode="pessimistic",
        SA_remove_kluge=False,
    )

Finally we can create the ``MapSim`` simulator object and pass the PySM custom component and the noise
simulator as dictionaries, we can also specify any default model from PySM as a comma separated string,
e.g. "d7,a1,s2"::

    simulator = mapsims.MapSim(
        telescope="SA",
        band=27,
        nside=NSIDE,
        unit="uK_CMB",
        pysm_components_string="a1",
        pysm_custom_components={"dust": dust, "synchrotron": sync},
        other_components={"noise": noise},
    )

and compute the output map using the ``execute`` method::

    output_map = simulator.execute()


Reference/API
=============

.. automodapi:: mapsims
