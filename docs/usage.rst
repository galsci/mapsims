*****
Usage
*****

Configuration file
==================

First you need to create a configuration file, see ``data/example_config.toml`` included in the package
or `in the repository <https://github.com/simonsobs/mapsims/blob/master/mapsims/data/example_config.toml>`_.

It first defines some global configuration options like output :math:`N_{side}`, the unit and the
channels, then has 2 subsections. They both define subsections with a ``class`` attribute that
specifies which object should be instantiated; all other arguments are passed into the class
constructor.

* The ``pysm_components`` subsection allows to choose any pre-existing PySM model and later add any custom class, for example one from ``so_pysm_models``.
* The ``other_components`` section instead includes models that generate a map to be summed after PySM has been executed, for example the noise simulation.

``channels`` supports both simulating the Simons Observatory channels at single frequencies or top-hat bandpasses.
If you specify channels named by the telescope and the frequency in GHz ``"SA_27"``, the simulations are performed at a single frequency. Instead if you specify one of the bandpasses, for example ``"LA_MFF1"``, the simulations are executed with top-hat bandpasses (10 equally spaced points within the band integrated with the Trapezoidal rule).
If you specify shortcuts for groups of channels, i.e. ``"all"``, ``"LA"`` or ``"SA"``, top-hat bandpasses are selected.

Simulate other instruments
==========================

A custom instrument can be defined by providing instrument parameters via a HDF5 file in a specified format, see :py:class:`MapSim` for details on the format.
Planck channels at single frequencies are embedded in the package, pass ``instrument_parameters="planck_deltabandpass"`` to select it. See also ``planck_deltabandpass.h5`` in the ``mapsims/data`` folder as an example of the format.

Example of accessing the file::

    In [1]: import h5py

    In [2]: f = h5py.File("planck_deltabandpass.h5")

    In [3]: f.keys()
    Out[3]: <KeysViewHDF5 ['030', '044', '070', '100', '143', '217', '353', '545', '857']>

    In [4]: f["143"].attrs
    Out[4]: <Attributes of HDF5 object at 46913457708992>

    In [5]: list(f["143"].attrs)
    Out[5]: ['band', 'center_frequency_GHz', 'fwhm_arcmin']

    In [6]: f["143"].attrs["center_frequency_GHz"]
    Out[6]: 142.876

mapsims_run
===========

``mapsims_run`` is a script included in the package, it can be used to execute pipelines described
in a configuration file in the terminal::

    mapsims_run example_config.toml

MapSims object
==============

Create the simulator object with::

    import mapsims
    simulator = mapsims.from_config("example_config.toml")

This returns a :py:class:`.MapSims` object, then you can
produce the output map with::

    output_map = simulator.execute()

Python classes
==============

Using instead the Python classes, we first need to create the custom component objects, as
an example we will use all defaults options::

    NSIDE = 16
    cmb = mapsims.SOPrecomputedCMB(
        num=0,
        nside=NSIDE,
        lensed=False,
        aberrated=False,
        has_polarization=True,
        cmb_set=0,
        cmb_dir="mapsims/tests/data",
        input_units="uK_CMB",
    )


Then we can create a :py:class:`.SONoiseSimulator`, the most important parameter is the scanning strategy,
it can be either "classical" or "opportunistic"::

    noise = mapsims.SONoiseSimulator(
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
    )

Finally we can create the :py:class:`.MapSim` simulator object and pass the PySM custom component and the noise
simulator as dictionaries, we can also specify any default model from PySM as a comma separated string,
e.g. "d7,a1,s2"::

    simulator = mapsims.MapSim(
        channels="all",
        nside=NSIDE,
        unit="uK_CMB",
        pysm_output_reference_frame="G",
        pysm_components_string="a1",
        pysm_custom_components={"cmb": cmb},
        other_components={"noise": noise},
    )

and compute the output map using the ``execute`` method::

    output_map = simulator.execute()

write instead directly output FITS maps to disk with::

    simulator.execute(write_outputs=True)
