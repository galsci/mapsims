Summary of Models
*******************

This page contains high-level description of the models available within the ``mapsims`` package,
this models are specific to the Simons Observatory, more general PySM models are instead
available in the ``so_pysm_models`` package, see `the documentation about those models <https://so-pysm-models.readthedocs.io/en/latest/so_pysm_models/models.html>`_.

20180822 noise power spectra and hitmaps
========================================

The :py:class:`SONoiseSimulator` class provides a wrapper to call the software that was released on 20180822
to simulate power spectra of the noise taking into account the expected performance of the whole experiment.
The noise simulator accepts a number of parameters to configure the simulation, see the documentation
of the class of the source code of `the noise spectra simulator included in the repository <https://github.com/simonsobs/mapsims/blob/master/mapsims/SO_Noise_Calculator_Public_20180822.py>`_.

It also includes low-resolution simulated relative hitmaps for the "classical" and the "opportunistic" scanning
strategies.

As an example::

    from mapsims import noise
    import healpy as hp
    noise_sim = noise.SONoiseSimulator(telescope="LA", band=27, nside=128)
    hp.mollview(noise_sim.hitmap, title="Relative hitmap")
    noise_map = m = noise_sim.simulate()
    hp.mollview(noise_map[1], min=-100, max=100, unit="uK_CMB", title="Q noise map LA 27")

Cosmic Microwave Background simulations
=======================================

The ``so_pysm_models`` package provides a generic :py:class:`so_pysm_models.PrecomputedAlms` PySM component that can load a set of :math:`a_{\ell m}` coefficients and generate a map at the requested :math:`N_{side}`.

``mapsims`` has 2 classes that derive from ``PrecomputedAlms``:

* the class :py:class:`SOPrecomputedCMB` provides a specific naming convention tailored to the Simons Observatory simulations that are already available
* the class :py:class:`SOStandalonePrecomputedCMB` is useful to simulate CMB only maps, in this case it is wasteful to use PySM because it first creates a map and then performs other 2 spherical harmonics transforms to apply the beam smoothing. This class instead keeps the input in spherical harmonics domain, first applies the beam and then returns a map. The :py:meth:`SOStandalonePrecomputedCMB.simulate` method gets a :py:class:`Channel` object and returns a map already convolved with the channel's beam.

Available Cosmic Microwave Background simulations
=================================================

**Lensed CMB**

Available at NERSC at: ``/global/project/projectdirs/sobs/v4_sims/mbs/cmb``

* Input theory spectrum and parameters: The input spectra are based on a best-fit Planck cosmology.  The unlensed power spectra are available at https://github.com/ACTCollaboration/actsims/blob/master/data/cosmo2017_10K_acc3_scalCls.dat.  The maps are generated at :math:`\ell_{max}` = 8000 but are saved only at :math:`\ell_{max}` = 5100 (to save disk space).
* The lensing (i.e. kappa or :math:`\phi`) maps are Gaussian random fields obtained with the same cosmology.   The lensing is done at one-arcminute resolution using the routine ``pixell.lensing.rand_map()``.  We pass separate random seeds to make the CMB and :math:`\phi` maps, meaning that there is no correlation between T and :math:`\phi` (or E and :math:`\phi`) arising from the ISW effect (this might have an impact for people studying primordial non-Gaussianity). The input :math:`\phi` :math:`a_{\ell m}` are available at ``/global/project/projectdirs/sobs/v4_sims/mbs/cmb/input_phi/`` and can be accessed using the :py:meth:`SOPrecomputedCMB.get_phi_alm` function.
* The lensed power spectra, which represent the lensed maps to a very good approximation are at https://github.com/ACTCollaboration/actsims/blob/master/data/cosmo2017_10K_acc3_lensedCls.dat
* There are no primordial B modes (and similarly no contribution from tensors to the T maps).
