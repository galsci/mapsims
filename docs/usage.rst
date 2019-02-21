*****
Usage
*****

Configuration file
==================

First you need to create a configuration file, see ``example_config.cfg`` included in the package
or `in the repository<https://github.com/simonsobs/mapsims/blob/master/mapsims/example_config.cfg>`_.

It first defines some global configuration options like output :math:`N_{side}`, the unit and the
channels, then has 2 subsections. They both define subsections with a ``class`` attribute that
specifies which object should be instantiated; all other arguments are passed into the class
constructor.

* The ``pysm_components`` subsection allows to choose any pre-existing PySM model and later add
any custom class, for example one from ``so_pysm_models``.
* The ``other_components`` section instead includes models that generate a map to be summed after
PySM has been executed, for example the noise simulation.

mapsims_run
===========

``mapsims_run`` is a script included in the package, it can be used to execute pipelines described
in a configuration file in the terminal::

    mapsims_run example_config.cfg

MapSims object
==============

Create the simulator object with::

    import mapsims
    simulator = mapsims.from_config("example_config.cfg")

produce the output map with::

    output_map = simulator.execute()

Python classes
==============

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
