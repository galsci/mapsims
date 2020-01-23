import os.path
import numpy as np
import healpy as hp
from astropy.utils import data
import warnings

import pysm.units as u

from . import SO_Noise_Calculator_Public_20180822 as so_noise
from so_models_v3 import SO_Noise_Calculator_Public_v3_1_1 as so_models
from . import so_utils
from . import Channel
from . import utils as mutils

sensitivity_modes = {"baseline": 1, "goal": 2}
one_over_f_modes = {"pessimistic": 0, "optimistic": 1}
telescope_seed_offset = {"LA": 0, "SA": 1000}
default_mask_value = {"healpix": hp.UNSEEN, "car": np.nan}


class SONoiseSimulator:
    def __init__(
        self,
        telescopes=["LA"],
        nside=None,
        shape=None,
        wcs=None,
        ell_max=None,
        seed=None,
        return_uK_CMB=True,
        sensitivity_mode="baseline",
        apply_beam_correction=True,
        apply_kludge_correction=True,
        scanning_strategy="classical",
        no_power_below_ell=None,
        LA_number_LF=1,
        LA_number_MF=4,
        LA_number_UHF=2,
        SA_years_LF=1,
        SA_one_over_f_mode="pessimistic",
        hitmap_version="v0.1",
    ):
        """Simulate noise maps for Simons Observatory

        Simulate the noise power spectrum in spherical harmonics domain and then generate a map
        in microK_CMB or microK_RJ (based on return_uK_CMB)

        In the constructor, this object calls the published 20180822 noise simulator and generates
        the expected noise power spectra for all channels.
        Then you need to call the `simulate` method with a channel identifier to create a simulated map.

        Parameters
        ----------

        telescopes : list of strings
            List of telescope identifiers, typically `LA` or `SA` for the large aperture
            and small aperture, respectively.
        nside : int
            nside of HEALPix map. If None, uses
            rectangular pixel geometry specified through shape and wcs.
        shape : tuple of ints
            shape of ndmap array (see pixell.enmap). Must also specify wcs.
        wcs : astropy.wcs.wcs.WCS instance
            World Coordinate System for geometry of map (see pixell.enmap). Must
            also specify shape.
        ell_max : int
            Maximum ell for the angular power spectrum, if not provided set to 3 * nside when using healpix
            or 10000 * (1.0 / pixel_height_arcmin) when using CAR, corresponding roughly to the Nyquist
            frequency.
        seed : int
            Numpy random seed, each band is going to get a different seed as seed + band + (1000 for SA)
        return_uK_CMB : bool
            True, output is in microK_CMB, False output is in microK_RJ
        sensitivity_mode : str
            Value should be threshold, baseline or goal to use predefined sensitivities
        apply_beam_correction : bool
            Include the effect of the beam in the noise angular power spectrum
        apply_kludge_correction : bool
            If True, reduce the hitcount by a factor of 0.85 to account for not-uniformity in the scanning
        scanning_strategy : str
            Choose between the available scanning strategy hitmaps "classical" or "opportunistic" or
            path to a custom hitmap; it will be normalized, so absolute hitcount does not matter
        no_power_below_ell : int
            The input spectra have significant power at low ell, we can zero that power specifying an integer
            :math:`\ell` value here. The power spectra at :math:`\ell < \ell_0` are set to zero.
        LA_number_LF : int
            Number of Low Frequency tubes in LAT
        LA_number_MF : int
            Number of Medium Frequency tubes in LAT
        LA_number_UHF : int
            Number of Ultra High Frequency tubes in LAT
        SA_years_LF : int
            Number of years for the Low Frequency detectors to be deployed on the Small Aperture telescopes
        SA_one_over_f_mode : {"pessimistic", "optimistic", "none"}
            Correlated noise performance of the detectors on the Small Aperture telescopes
        hitmap_version : string
            Version string for hitmaps stored remotely.
        """

        if nside is None:
            assert shape is not None
            assert wcs is not None
            self.healpix = False
            self.shape = shape
            self.wcs = wcs
            self._pixheight = np.abs(wcs.wcs.cdelt[0] * 60.0)
            self.ell_max = (
                ell_max if ell_max is not None else 10000 * (1.0 / self._pixheight)
            )
        else:
            assert shape is None
            assert wcs is None
            self.healpix = True
            self.nside = nside
            self.ell_max = ell_max if ell_max is not None else 3 * nside

        self.sensitivity_mode = sensitivity_modes[sensitivity_mode]
        self.apply_beam_correction = apply_beam_correction
        self.apply_kludge_correction = apply_kludge_correction
        self.seed = seed
        self.return_uK_CMB = return_uK_CMB
        self.no_power_below_ell = no_power_below_ell
        self.LA_number_LF = LA_number_LF
        self.LA_number_MF = LA_number_MF
        self.LA_number_UHF = LA_number_UHF
        self.SA_years_LF = SA_years_LF
        self.SA_one_over_f_mode = one_over_f_modes[SA_one_over_f_mode]

        self.remote_data = mutils.RemoteData(
            healpix=self.healpix, version=hitmap_version
        )

        # Load hitmap and compute sky fraction

        self.hitmap = {}
        self.sky_fraction = {}

        self.noise_ell_T = {"SA": {}, "LA": {}}
        self.noise_ell_P = {"SA": {}, "LA": {}}
        self.ch = []
        for telescope in telescopes:
            self.update_telescope(telescope, scanning_strategy)

    def update_telescope(self, telescope, scanning_strategy):
        """Update a telescope configuration by loading the corresponding
        hitmaps. Each loaded `telescope` is kept in memory, but
        new choice of `scanning_strategy` erases the previous one.

        Parameters
        ----------

        telescope : string
            Telescope identifier, typically `LA` or `SA` for the large aperture
            and small aperture, respectively.
        scanning_strategy : str
            Choose between the available scanning strategy hitmaps "classical" or "opportunistic" or
            path to a custom hitmap; it will be normalized, so absolute hitcount does not matter

        """

        if not (self.healpix):
            npixheight = min(
                {"LA": [0.5, 2.0], "SA": [4.0, 12.0]}[telescope],
                key=lambda x: abs(x - self._pixheight),
            )
            car_suffix = f"_CAR_{npixheight:.2f}_arcmin"
        else:
            car_suffix = ""

        if os.path.exists(scanning_strategy.format(telescope=telescope)):
            hitmap_filename = scanning_strategy
        else:
            rname = f"total_hits_{telescope}_{scanning_strategy}{car_suffix}.fits.gz"
            hitmap_filename = self.remote_data.get(rname)

        if self.healpix:
            hitmap = hp.ud_grade(
                hp.read_map(hitmap_filename, verbose=False), nside_out=self.nside
            )
        else:
            from pixell import enmap, wcsutils

            hitmap = enmap.read_map(hitmap_filename)
            if wcsutils.is_compatible(hitmap.wcs, self.wcs):
                hitmap = enmap.extract(hitmap, self.shape, self.wcs)
            else:
                warnings.warn(
                    "WCS of hitmap with nearest pixel-size is not compatible, so interpolating hitmap"
                )
                hitmap = enmap.project(hitmap, self.shape, self.wcs)

        hitmap /= hitmap.max()
        # Discard pixels with very few hits that cause border effects
        # hitmap[hitmap < 1e-3] = 0
        self.hitmap[telescope] = hitmap
        self.sky_fraction[telescope] = (hitmap != 0).sum() / hitmap.size

        if telescope == "SA":
             survey = so_models.SOSatV3point1(
                sensitivity_mode=self.sensitivity_mode,
                survey_efficiency=1,
                survey_years=self.SA_years_LF,
                N_tubes=None,  # FIXME expose to configuration
                el=None,  # FIXME expose to configuration
                one_over_f_mode=self.SA_one_over_f_mode,
             )
             ell, noise_ell_T, noise_ell_P = survey.get_noise_curves(
                self.sky_fraction[telescope],
                self.ell_max,
                delta_ell=1,
                full_covar=False,
                deconv_beam=self.apply_beam_correction,
             )
            # For SA, so_noise simulates only Polarization,
            # Assume that T is half
             if noise_ell_T is None:
                noise_ell_T = noise_ell_P / 2
            ell, noise_ell_P, _ = so_noise.Simons_Observatory_V3_SA_noise(
                self.sensitivity_mode,
                self.SA_one_over_f_mode,
                self.SA_years_LF,
                self.sky_fraction[telescope],
                self.ell_max,
                delta_ell=1,
                apply_beam_correction=self.apply_beam_correction,
                apply_kludge_correction=self.apply_kludge_correction,
            )
            # For SA, so_noise simulates only Polarization,
            # Assume that T is half
            noise_ell_T = noise_ell_P / 2
        elif telescope == "LA":
            ell, noise_ell_T, noise_ell_P, _ = so_noise.Simons_Observatory_V3_LA_noise(
                self.sensitivity_mode,
                self.sky_fraction[telescope],
                self.ell_max,
                delta_ell=1,
                N_LF=self.LA_number_LF,
                N_MF=self.LA_number_MF,
                N_UHF=self.LA_number_UHF,
                apply_beam_correction=self.apply_beam_correction,
                apply_kludge_correction=self.apply_kludge_correction,
            )
        self.ell = np.arange(ell[-1] + 1)

        available_frequencies = np.unique(so_utils.frequencies)
        for frequency in so_utils.frequencies:
            band_index = available_frequencies.searchsorted(frequency)

            # so_noise returns power spectrum starting with ell=2, start instead at 0
            # repeat the value at ell=2 for lower multipoles
            self.noise_ell_T[telescope][frequency] = np.zeros(
                len(self.ell), dtype=np.double
            )
            self.noise_ell_P[telescope][frequency] = self.noise_ell_T[telescope][
                frequency
            ].copy()
            self.noise_ell_T[telescope][frequency][2:] = noise_ell_T[band_index]
            self.noise_ell_T[telescope][frequency][:2] = 0
            self.noise_ell_P[telescope][frequency][2:] = noise_ell_P[band_index]
            self.noise_ell_P[telescope][frequency][:2] = 0

            if self.no_power_below_ell is not None:
                self.noise_ell_T[telescope][frequency][
                    self.ell < self.no_power_below_ell
                ] = 0
                self.noise_ell_P[telescope][frequency][
                    self.ell < self.no_power_below_ell
                ] = 0

    def simulate(
        self, ch, output_units="uK_CMB", seed=None, nsplits=1, mask_value=None
    ):
        """Create a random realization of the noise power spectrum

        Parameters
        ----------

        ch : mapsims.Channel
            Channel identifier, create with e.g. mapsims.SOChannel("SA", 27)
        nsplits : integer, optional
            Number of splits to generate. The splits will have independent noise
            realizations, with noise power scaled by a factor of nsplits. By default,
            only one split (the coadd) is generated.
        mask_value : float, optional
            The value to set in masked (unobserved) regions. By default, it uses
            the value in default_mask_value, which for healpix is healpy.UNSEEN
            and for CAR is numpy.nan.

        Returns
        -------

        output_map : ndarray
            Numpy array with the HEALPix map realization of noise
        """
        assert nsplits >= 1
        if seed is not None:
            np.random.seed(seed)
        else:
            if self.seed is not None:
                try:
                    frequency_offset = int(ch.band)
                except ValueError:
                    frequency_offset = so_utils.bands.index(ch.band) * 100
                np.random.seed(
                    self.seed + frequency_offset + telescope_seed_offset[ch.telescope]
                )
        zeros = np.zeros_like(self.noise_ell_T[ch.telescope][ch.center_frequency.value])
        ps = (
            np.asarray(
                [
                    self.noise_ell_T[ch.telescope][ch.center_frequency.value],
                    self.noise_ell_P[ch.telescope][ch.center_frequency.value],
                    self.noise_ell_P[ch.telescope][ch.center_frequency.value],
                    zeros,
                    zeros,
                    zeros,
                ]
            )
            * nsplits
        )
        if self.healpix:
            npix = hp.nside2npix(self.nside)
            output_map = np.zeros((nsplits, 3, npix))
            for i in range(nsplits):
                output_map[i] = hp.ma(
                    np.array(
                        hp.synfast(
                            ps, nside=self.nside, pol=True, new=True, verbose=False,
                        )
                    )
                )
        else:
            from pixell import curvedsky, powspec

            ps = powspec.sym_expand(np.asarray(ps), scheme="diag")
            output_map = np.zeros((nsplits, 3) + self.shape)
            for i in range(nsplits):
                output_map[i] = curvedsky.rand_map((3,) + self.shape, self.wcs, ps)

        hmap = self.hitmap[ch.telescope]
        good = hmap != 0
        # Normalize on the Effective sky fraction, see discussion in:
        # https://github.com/simonsobs/mapsims/pull/5#discussion_r244939311
        output_map[:, :, good] /= np.sqrt(
            hmap[good] / hmap.mean() * self.sky_fraction[ch.telescope]
        )
        if mask_value is None:
            mask_value = (
                default_mask_value["healpix"]
                if self.healpix
                else default_mask_value["car"]
            )
        output_map[:, :, np.logical_not(good)] = mask_value
        unit_conv = (1 * u.uK_CMB).to_value(
            u.Unit(output_units), equivalencies=u.cmb_equivalencies(ch.center_frequency)
        )
        output_map *= unit_conv
        return output_map[0] if nsplits == 1 else output_map
