import numpy as np
import healpy as hp

import pysm

from . import SO_Noise_Calculator_Public_20180822 as so_noise

sensitivity_modes = {"baseline": 1, "goal": 2}
one_over_f_modes = {"pessimistic": 0, "optimistic": 1}


class SONoiseSimulator:

    def __init__(
        self,
        telescope,
        band,
        nside,
        return_K_CMB=True,
        sensitivity_mode="baseline",
        apply_beam_correction=False,
        scanning_strategy="classical",
        LA_number_LF=1,
        LA_number_MF=4,
        LA_number_UHF=4,
        SA_years_LF=None,
        SA_one_over_f_mode=None,
        SA_remove_kluge=False,
    ):
        """Simulate noise maps for Simons Observatory

        Simulate the noise power spectrum in spherical harmonics domain and then generate a map

        Parameters
        ----------

        telescope : {"SA", "LA"}
            Either LA or SA for Large and Small aperture
        band : int
            Detectors frequency band in GHz
        nside : int
            Output HEALPix NSIDE
        return_K_CMB : bool
            True, output is in K_CMB, False output is in K_RJ
        sensitivity_mode : str
            Value should be threshold, baseline or goal to use predefined sensitivities
        apply_beam_correction : bool
            Include the effect of the beam in the noise angular power spectrum
        scanning_strategy : {"classical", "opportunistic"}
            Choose between the available scanning strategy hitmaps
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
        self.scanning_strategy = scanning_strategy
        assert scanning_strategy in ["classical", "opportunistic"]
        self.nside = nside
        self.return_K_CMB = return_K_CMB
        self.ell_max = 3 * nside
        self.LA_number_LF = LA_number_LF
        self.LA_number_MF = LA_number_MF
        self.LA_number_UHF = LA_number_UHF
        self.SA_years_LF = SA_years_LF
        self.SA_one_over_f_mode = one_over_f_modes[SA_one_over_f_mode]
        self.SA_remove_kluge = SA_remove_kluge

        if self.telescope == "SA":
            self.ell, self.noise_ell_P, _ = so_noise.Simons_Observatory_V3_SA_noise(
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
            self.noise_ell_T = self.noise_ell_P / 2
        elif self.telescope == "LA":
            self.ell, self.noise_ell_T, self.noise_ell_P, _ = so_noise.Simons_Observatory_V3_LA_noise(
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

        bands = getattr(so_noise, "so_V3_{}_bands".format(self.telescope)()).astype(
            np.int
        )
        band_index = bands.searchsorted(self.band)
        self.noise_ell_T = self.noise_ell_T[band_index]
        self.noise_ell_P = self.noise_ell_P[band_index]

        if self.return_K_CMB:
            to_K_CMB = pysm.convert_units("K_RJ", "K_CMB", band) ** 2
            self.noise_ell_T *= to_K_CMB
            self.noise_ell_P *= to_K_CMB

    def simulate(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        zeros = np.zeros_like(self.noise_ell_T)
        output_map = hp.synfast(
            [self.noise_ell_T, self.noise_ell_P, self.noise_ell_P, zeros, zeros, zeros],
            nside=self.nside,
            pol=True,
            new=True,
            verbose=False,
        )
        return output_map
