Summary of Models
*******************

This page contains high-level description of the models available within the ``mapsims`` package,
this models are specific to the Simons Observatory, more general PySM models are instead
available in the ``so_pysm_models`` package, see `the documentation about those models <https://so-pysm-models.readthedocs.io/en/latest/models.html>`_.

Model templates
===============

Model templates or more in general model data are stored on the Simons Observatory project pages at NERSC in the folder::

    /global/project/projectdirs/sobs/www/so_mapsims_data

Which is then made publicly available via web at https://portal.nersc.gov/project/sobs/so_mapsims_data

For example all the available hitmaps are available there. In order to copy data at that location:

* run `collabsu sobs` (you need permissions by the allocation PI)
* copy data there
* run `bash update_html_list.sh` in the `/global/project/projectdirs/sobs/www/so_mapsims_data` folder, this creates the html pages that list the files and also fixes permissions.

Noise power spectra and hitmaps
===============================

The :py:class:`.SONoiseSimulator` class provides a wrapper to call `so_noise_models`
to simulate power spectra of the noise taking into account the expected performance of the whole experiment.
The noise simulator accepts a number of parameters to configure the simulation, see the documentation
in the `so_noise_models repository <https://github.com/simonsobs/so_noise_models>`_.

It also includes simulated relative hitmaps simulated in time domain with TOAST.
We currently have 1 hitmap per tube and they are the first split of the `MSS-0001` Mission Scale simulation (~18 days of data), see `the wiki (restricted to members of Simons Observatory) for more documentation <http://simonsobservatory.wikidot.com/mss-0001>`_.

As an example::

    >>> from mapsims import noise
    >>> import healpy as hp
    >>> noise_sim = noise.SONoiseSimulator(nside=128)
    >>> hp.mollview(noise_sim.get_hitmaps("ST0")[0][0], title="Relative hitmap")
    >>> noise_maps = noise_sim.simulate(tube="ST0")
    >>> hp.mollview(noise_maps[0][0][1], min=-10, max=10, unit="uK_CMB", title="Q noise map ST0")

Cosmic Microwave Background simulations
=======================================

The ``so_pysm_models`` package provides a generic :py:class:`so_pysm_models.PrecomputedAlms` PySM component that can load a set of :math:`a_{\ell m}` coefficients and generate a map at the requested :math:`N_{side}`.

``mapsims`` has 2 classes that derive from ``PrecomputedAlms``:

* the class :py:class:`.SOPrecomputedCMB` provides a specific naming convention tailored to the Simons Observatory simulations that are already available
* the class :py:class:`.SOStandalonePrecomputedCMB` is useful to simulate CMB only maps, in this case it is wasteful to use PySM because it first creates a map and then performs other 2 spherical harmonics transforms to apply the beam smoothing. This class instead keeps the input in spherical harmonics domain, first applies the beam and then returns a map. The :py:meth:`.SOStandalonePrecomputedCMB.simulate` method gets a :py:class:`.Channel` object and returns a map already convolved with the channel's beam.

Available Cosmic Microwave Background simulations
=================================================

**Lensed CMB**

Available at NERSC at: ``/global/project/projectdirs/sobs/v4_sims/mbs/cmb``

* Input theory spectrum and parameters: The input spectra are based on a best-fit Planck cosmology.  The unlensed power spectra are available at https://github.com/ACTCollaboration/actsims/blob/master/data/cosmo2017_10K_acc3_scalCls.dat.  The maps are generated at :math:`\ell_{max}` = 8000 but are saved only at :math:`\ell_{max}` = 5100 (to save disk space).
* The lensing (i.e. kappa or :math:`\phi`) maps are Gaussian random fields obtained with the same cosmology.   The lensing is done at one-arcminute resolution using the routine ``pixell.lensing.rand_map()``.  We pass separate random seeds to make the CMB and :math:`\phi` maps, meaning that there is no correlation between T and :math:`\phi` (or E and :math:`\phi`) arising from the ISW effect (this might have an impact for people studying primordial non-Gaussianity). The input :math:`\phi` :math:`a_{\ell m}` are available at ``/global/project/projectdirs/sobs/v4_sims/mbs/cmb/input_phi/`` and can be accessed using the :py:meth:`.SOPrecomputedCMB.get_phi_alm` function.
* The lensed power spectra, which represent the lensed maps to a very good approximation are at https://github.com/ACTCollaboration/actsims/blob/master/data/cosmo2017_10K_acc3_lensedCls.dat
* There are no primordial B modes (and similarly no contribution from tensors to the T maps).
* See all the cosmological parameters used in the :doc:`CAMB configuration file <camb>`
