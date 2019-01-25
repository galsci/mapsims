import os.path
import numpy as np
import healpy as hp
from astropy.utils import data

import pysm

from . import SO_Noise_Calculator_Public_20180822 as so_noise
from .so_utils import get_bands
from . import Channel

sensitivity_modes = {"baseline": 1, "goal": 2}
one_over_f_modes = {"pessimistic": 0, "optimistic": 1}
telescope_seed_offset = {"LA": 0, "SA": 1000}


class SONoiseSimulator:

    def __init__(
        self,
        nside,
        ell_max=None,
        seed=None,
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
    ):
        """Simulate noise maps for Simons Observatory

        Simulate the noise power spectrum in spherical harmonics domain and then generate a map
        in microK_CMB or microK_RJ (based on return_uK_CMB)

        In the constructor, this object calls the published 20180822 noise simulator and generates
        the expected noise power spectra for all channels.
        Then you need to call the `simulate` method with a channel identifier to create a simulated map.

        Parameters
        ----------

        nside : int
            Output HEALPix NSIDE
        ell_max : int
            Maximum ell for the angular power spectrum, if not provided set to 3 * nside
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
            path to a custom hitmap, it will be normalized, absolute hitcount does not matter
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
        """

        self.sensitivity_mode = sensitivity_modes[sensitivity_mode]
        self.apply_beam_correction = apply_beam_correction
        self.apply_kludge_correction = apply_kludge_correction
        self.nside = nside
        self.seed = seed
        self.return_uK_CMB = return_uK_CMB
        self.ell_max = ell_max if ell_max is not None else 3 * nside
        self.LA_number_LF = LA_number_LF
        self.LA_number_MF = LA_number_MF
        self.LA_number_UHF = LA_number_UHF
        self.SA_years_LF = SA_years_LF
        self.SA_one_over_f_mode = one_over_f_modes[SA_one_over_f_mode]

        # Load hitmap and compute sky fraction

        self.hitmap = {}
        self.sky_fraction = {}
        self.noise_ell_T = {}
        self.noise_ell_P = {}
        for telescope in ["LA", "SA"]:
            if os.path.exists(scanning_strategy.format(telescope=telescope)):
                hitmap_filename = scanning_strategy
            else:
                hitmap_filename = data.get_pkg_data_filename(
                    "data/total_hits_{}_{}.fits.gz".format(telescope, scanning_strategy)
                )
            hitmap = hp.ud_grade(
                hp.read_map(hitmap_filename, verbose=False), nside_out=self.nside
            )
            hitmap /= hitmap.max()
            # Discard pixels with very few hits that cause border effects
            # hitmap[hitmap < 1e-3] = 0
            self.hitmap[telescope] = hitmap
            self.sky_fraction[telescope] = (hitmap != 0).sum() / len(hitmap)

            if telescope == "SA":
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

            for band_index, band in enumerate(get_bands(telescope)):

                ch = Channel(telescope, band)

                # so_noise returns power spectrum starting with ell=2, start instead at 0
                # repeat the value at ell=2 for lower multipoles
                self.noise_ell_T[ch] = np.zeros(len(self.ell), dtype=np.double)
                self.noise_ell_P[ch] = self.noise_ell_T[ch].copy()
                self.noise_ell_T[ch][2:] = noise_ell_T[band_index]
                self.noise_ell_T[ch][:2] = 0
                self.noise_ell_P[ch][2:] = noise_ell_P[band_index]
                self.noise_ell_P[ch][:2] = 0

                if not self.return_uK_CMB:
                    to_K_RJ = pysm.convert_units("K_CMB", "K_RJ", band) ** 2
                    self.noise_ell_T[ch] *= to_K_RJ
                    self.noise_ell_P[ch] *= to_K_RJ

    def simulate(self, ch):
        """Create a random realization of the noise power spectrum

        Parameters
        ----------

        ch : mapsims.Channel
            Channel identifier, create with e.g. mapsims.Channel("SA", 27)

        Returns
        -------

        output_map : ndarray
            Numpy array with the HEALPix map realization of noise
        """
        if self.seed is not None:
            np.random.seed(self.seed + ch.band + telescope_seed_offset[ch.telescope])
        zeros = np.zeros_like(self.noise_ell_T[ch])
        output_map = hp.ma(
            np.array(
                hp.synfast(
                    [
                        self.noise_ell_T[ch],
                        self.noise_ell_P[ch],
                        self.noise_ell_P[ch],
                        zeros,
                        zeros,
                        zeros,
                    ],
                    nside=self.nside,
                    pol=True,
                    new=True,
                    verbose=False,
                )
            )
        )
        good = self.hitmap[ch.telescope] != 0
        # Normalize on the Effective sky fraction, see discussion in:
        # https://github.com/simonsobs/mapsims/pull/5#discussion_r244939311
        output_map[:, good] /= np.sqrt(
            self.hitmap[ch.telescope][good]
            / self.hitmap[ch.telescope].mean()
            * self.sky_fraction[ch.telescope]
        )
        output_map[:, np.logical_not(good)] = hp.UNSEEN
        return output_map
