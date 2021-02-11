*****
Usage
*****

Configuration file
==================

First you need to create a configuration file, see ``data/example_config_v0.2.toml`` included in the package
or `in the repository <https://github.com/simonsobs/mapsims/blob/master/mapsims/data/example_config_v0.2.toml>`_.

It first defines some global configuration options like output :math:`N_{side}`, the desired output unit and the
channels, and a tag to define the output filenames, then has 2 sections. They both define subsections with a ``class`` attribute that
specifies which object should be instantiated; all other arguments are passed into the class
constructor.

* The ``pysm_components`` subsection allows to choose any pre-existing PySM model (using ``pysm_components_string`` and ``pysm_output_reference_frame``) and later add any custom class, for example one from ``so_pysm_models``.
* The ``other_components`` section instead includes models that generate a map to be summed after PySM has been executed, for example the noise simulation.

All the arguments to the different components are defined by each class, the most general components are defined in ``so_pysm_models``, see `the documentation <https://so-pysm-models.readthedocs.io/en/latest/models.html>`_, for example :py:class:`so_pysm_models.WebSkyCIB`, the models specific to Simons Observatory are instead defined directly in ``mapsims``, for example :py:class:`SONoiseSimulator`.
Another option is to look through the available simulations in the `Map-based simulations repository <https://github.com/simonsobs/map_based_simulations>`_ and inspect the ``toml`` configuration files that were used for previous simulations.

``channels`` supports both simulating the Simons Observatory channels with top-hat bandpasses.
If you specify channels named by the tube and the band name, for example ``"ST3_LF2"``, the simulations are executed with top-hat bandpasses (10 equally spaced points within the band integrated with the Trapezoidal rule).
If you specify shortcuts for groups of channels, i.e. ``"all"``, ``"telescope:LA"`` or ``"telescope:SA"``.
We also support simulating a dichroic tube which includes also the full covariance due to the atmosphere, in this case you can set channels to a tube tag, e.g. ``"tube:ST3"`` for the Small Aperture telescope tube 3 which includes the ``LF1`` and ``LF2`` bands.

The simulation seed
-------------------

All the components which are not deterministic accept a configuration option ``num`` that sets the seeds for the random number generator in order to being able to reproduce the exact same simulation later on, for example :py:class:`SONoiseSimulator`. Or they
pre-load a specific realization of simulations that were previously executed, for example :py:class:`SOPrecomputedCMB`..
The components also automatically apply a shift on the seed based on the channel that you are simulating so that the seeds for the channels are always different.

``mapsims`` configuration files have a **global simulation number** ``num`` defined
in the top level of the configuration file (or the constructor of :py:class:`MapSim`).
This number is automatically also passed to all the different components so that it uniquely identifies
a specific simulation.
You are allowed to override this by also setting the ``num`` parameter separately in the component classes.

Simulate other instruments
==========================

A custom instrument can be defined by providing instrument parameters via a IPAC text table file in a specified format, see :py:class:`MapSim` for details on the format.
Planck channels at single frequencies are embedded in the package, pass ``instrument_parameters="planck_deltabandpass"`` to select it. See also ``planck_deltabandpass.tbl`` in the ``mapsims/data/planck_deltabandpass`` folder as an example of the format.

Example of accessing the file::

	In [1]: from astropy.table import QTable

	In [2]: f = QTable.read("planck_deltabandpass/planck_deltabandpass.tbl", format="ascii.ipac")

	In [3]: f.colnames
	Out[3]: ['band', 'center_frequency', 'fwhm']

	In [4]: f
	Out[4]: 
	<QTable length=9>
	band center_frequency     fwhm    
			   GHz           arcmin   
	str3     float64        float64   
	---- ---------------- ------------
	  30             28.4 33.102652125
	  44             44.1  27.94348615
	  70             70.4  13.07645961
	 100           100.89        9.682
	 143          142.876        7.303
	 217          221.156        5.021
	 353            357.5        4.944
	 545            555.2        4.831
	 857            866.8        4.638

	In [10]: f[0]["fwhm"] # access by row index
	Out[10]: <Quantity 33.10265212 arcmin>

	In [11]: f.add_index("band")

	In [12]: f.loc["70"]["center_frequency"] # access by tag (str not integer)
	Out[12]: <Quantity 70.4 GHz>


mapsims_run
===========

``mapsims_run`` is a script included in the package, it can be used to execute pipelines described
in a configuration file in the terminal and write the output to FITS files::

    mapsims_run example_config_v0.2.toml

It also supports overriding from the command line a subset of the parameters, here the full list::

    mapsims_run --nside 32 --channels tube:ST1 --num 4 example_config_v0.2.toml

MapSims object
==============

Create the simulator object with::

    import mapsims
    simulator = mapsims.from_config("example_config_v0.2.toml")

This returns a :py:class:`.MapSims` object, then you can
produce the output maps with::

    output_maps = simulator.execute()

Python classes
==============

Using instead the Python classes, we first need to create the custom component objects, as
an example we will use all defaults options::

    >>> import mapsims
    >>> NSIDE = 16
    >>> cmb = mapsims.SOPrecomputedCMB(
    ...     num=0,
    ...     nside=NSIDE,
    ...     lensed=False,
    ...     aberrated=False,
    ...     has_polarization=True,
    ...     cmb_set=0,
    ...     cmb_dir="mapsims/tests/data",
    ...     input_units="uK_CMB",
    ... )


Then we can create a :py:class:`.SONoiseSimulator`, the most important parameter is the scanning strategy,
it can be either "classical" or "opportunistic"::

    >>> noise = mapsims.SONoiseSimulator(
    ...     nside=NSIDE,
    ...     return_uK_CMB=True,
    ...     sensitivity_mode="baseline",
    ...     apply_beam_correction=True,
    ...     apply_kludge_correction=True,
    ...     SA_one_over_f_mode="pessimistic",
    ... )

Finally we can create the :py:class:`.MapSim` simulator object and pass the PySM custom component and the noise
simulator as dictionaries, we can also specify any default model from PySM as a comma separated string,
e.g. "d7,a1,s2"::

    >>> simulator = mapsims.MapSim(
    ...     channels="tube:ST0",
    ...     nside=NSIDE,
    ...     unit="uK_CMB",
    ...     pysm_output_reference_frame="G",
    ...     pysm_components_string="a1",
    ...     pysm_custom_components={"cmb": cmb},
    ...     other_components={"noise": noise},
    ... )

and compute the output map using the ``execute`` method::

    output_map = simulator.execute()

write instead directly output FITS maps to disk with::

    simulator.execute(write_outputs=True)
