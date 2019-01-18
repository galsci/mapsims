import os.path
import numpy as np
import healpy as hp
from astropy.utils import data

import pysm

from . import SO_Noise_Calculator_Public_20180822 as so_noise

sensitivity_modes = {"baseline": 1, "goal": 2}
one_over_f_modes = {"pessimistic": 0, "optimistic": 1}


def get_bands(telescope):
    """Returns the available bands for a telescope

    Parameters
    ----------
    telescope : {"SA", "LA"}

    Returns
    -------
    bands : ndarray of ints
        Available bands
    """
    bands = getattr(
        so_noise, "Simons_Observatory_V3_{}_bands".format(telescope)
    )().astype(np.int)
    return bands


class SONoiseSimulator:

    def __init__(
        self,
        telescope,
        band,
        nside,
        ell_max=None,
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

        Parameters
        ----------

        telescope : {"SA", "LA"}
            Either LA or SA for Large and Small aperture
        band : int
            Detectors frequency band in GHz
        nside : int
            Output HEALPix NSIDE
        ell_max : int
            Maximum ell for the angular power spectrum, if not provided set to 3 * nside
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

        self.telescope = telescope
        assert telescope in ["LA", "SA"]
        self.band = int(band)
        self.sensitivity_mode = sensitivity_modes[sensitivity_mode]
        self.apply_beam_correction = apply_beam_correction
        self.apply_kludge_correction = apply_kludge_correction
        self.nside = nside
        self.return_uK_CMB = return_uK_CMB
        self.ell_max = ell_max if ell_max is not None else 3 * nside
        self.LA_number_LF = LA_number_LF
        self.LA_number_MF = LA_number_MF
        self.LA_number_UHF = LA_number_UHF
        self.SA_years_LF = SA_years_LF
        self.SA_one_over_f_mode = one_over_f_modes[SA_one_over_f_mode]

        # Load hitmap and compute sky fraction

        if os.path.exists(scanning_strategy):
            hitmap_filename = scanning_strategy
        else:
            hitmap_filename = data.get_pkg_data_filename(
                "data/total_hits_{}_{}.fits.gz".format(
                    self.telescope, scanning_strategy
                )
            )
        self.hitmap = hp.ud_grade(
            hp.read_map(hitmap_filename, verbose=False), nside_out=self.nside
        )
        self.hitmap /= self.hitmap.max()
        # Discard pixels with very few hits that cause border effects
        # self.hitmap[self.hitmap < 1e-3] = 0
        self.sky_fraction = (self.hitmap != 0).sum() / len(self.hitmap)

        if self.telescope == "SA":
            ell, noise_ell_P, _ = so_noise.Simons_Observatory_V3_SA_noise(
                self.sensitivity_mode,
                self.SA_one_over_f_mode,
                self.SA_years_LF,
                self.sky_fraction,
                self.ell_max,
                delta_ell=1,
                apply_beam_correction=self.apply_beam_correction,
                apply_kludge_correction=self.apply_kludge_correction,
            )
            # For SA, so_noise simulates only Polarization,
            # Assume that T is half
            noise_ell_T = noise_ell_P / 2
        elif self.telescope == "LA":
            ell, noise_ell_T, noise_ell_P, _ = so_noise.Simons_Observatory_V3_LA_noise(
                self.sensitivity_mode,
                self.sky_fraction,
                self.ell_max,
                delta_ell=1,
                N_LF=self.LA_number_LF,
                N_MF=self.LA_number_MF,
                N_UHF=self.LA_number_UHF,
                apply_beam_correction=self.apply_beam_correction,
                apply_kludge_correction=self.apply_kludge_correction,
            )

        # extract the relevant band
        bands = get_bands(self.telescope)
        try:
            band_index = bands.tolist().index(self.band)
        except ValueError:
            print(
                "Band {} not available, available bands for {} are {}".format(
                    self.band, self.telescope, bands
                )
            )
            raise

        # so_noise returns power spectrum starting with ell=2, start instead at 0
        # repeat the value at ell=2 for lower multipoles
        self.ell = np.arange(ell[-1]+1)
        self.noise_ell_T = np.zeros(len(self.ell), dtype=np.double)
        self.noise_ell_P = self.noise_ell_T.copy()
        self.noise_ell_T[2:] = noise_ell_T[band_index]
        self.noise_ell_T[:2] = self.noise_ell_T[2]
        self.noise_ell_P[2:] = noise_ell_P[band_index]
        self.noise_ell_P[:2] = self.noise_ell_P[2]

        if not self.return_uK_CMB:
            to_K_RJ = pysm.convert_units("K_CMB", "K_RJ", band) ** 2
            self.noise_ell_T *= to_K_RJ
            self.noise_ell_P *= to_K_RJ

    def simulate(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        zeros = np.zeros_like(self.noise_ell_T)
        output_map = hp.ma(
            np.array(
                hp.synfast(
                    [
                        self.noise_ell_T,
                        self.noise_ell_P,
                        self.noise_ell_P,
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
        good = self.hitmap != 0
        # see explanation of the normalization at:
        # https://github.com/simonsobs/mapsims/pull/5#discussion_r244939311
        output_map[:, good] /= np.sqrt(
            self.hitmap[good] * self.hitmap.mean() / self.sky_fraction
        )
        output_map[:, np.logical_not(good)] = hp.UNSEEN
        return output_map
