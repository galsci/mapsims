from scipy.interpolate import interp1d
import numpy as np


try:  # PySM >= 3.2.1
    import pysm3.units as u
except ImportError:
    import pysm.units as u

import mapsims.noise

# An example simple survey with external noise curves and hitmaps.
class SurveyFromExternalData:
    def __init__(
        self,
        nbands,
        fwhms,
        noise_ell=None,
        noise_TT=None,
        noise_PP=None,
        noise_files=None,
        hitmaps=None,
        hitmap_filenames=None,
        ivar_maps=None,
        ivar_map_filenames=None,
        white_noises=None,
    ):

        """[summary]

        [description]

        Arguments:
            nbands {[int]} -- the number of bands to be passed
            fwhms {[float]} -- a list of the fwhms for each band
        Keyword Arguments:

            Supply either:
            noise_files {[type]} -- a list of the noise files for each band. The noise will then be loaded from files
            or
            noise_ell,noise_TT and noise_PP -- noise_ell is a 1d array of the noise power spectra ells.
                    noise_TT and noise_PP are the arrays of noise for each band. They should have shape
                    either [nband,len(noise_ell)] or [nband,nband,len(noise_ell)] if band covariances are desired.


            Supply either
            hitmaps {array} -- an array of [nbands,map dimensions] containing the hitmaps for each band
            or
            hitmap_filenames {[type]} -- a list of files to load the hitmap for each band

            Supply either:

            ivar_maps {[type]} -- an array of [nbands,map dimensions] containing ivar_maps for each band
            or

            ivar_map_filenames {[type]} -- a list of filenames for the ivar_maps for each band(default: {None})


            white_noises {[type]} -- a list of the white noise levels for each band. Note this is not necessary for basic functionality
                                    However if not passed some functions of the noise class will not work. E.g. rescaling the white noise level
                                    of generating an ivar_map (if neither of the previous two arguments are supplied.
        """
        self.nbands = nbands
        self.fwhms = np.array(fwhms)

        self.noise_files = noise_files
        self.noise_ell = noise_ell
        self.noise_TT = noise_TT
        self.noise_PP = noise_PP

        self.hitmaps = hitmaps
        self.hitmap_filenames = hitmap_filenames
        self.ivar_maps = ivar_maps
        self.ivar_map_filenames = ivar_map_filenames

        self.white_noises = white_noises

        # Check that there is an input for each band.
        if np.shape(fwhms)[0] != nbands:
            raise AssertionError("Each band needs a fwhm")
        if noise_files is not None:
            if np.shape(noise_files)[0] != nbands:
                raise AssertionError("Each band needs a noisefile")
        else:
            if np.shape(noise_TT)[0] != nbands or np.shape(noise_PP)[0] != nbands:
                print(np.shape(noise_TT)[0], noise_PP.shape[0], nbands)
                raise AssertionError("Each band needs a noise spectrum")

        if hitmap_filenames is not None:
            if np.shape(hitmap_filenames)[0] != nbands:
                raise AssertionError("Each band needs a hitmap_filename")
        elif hitmaps is not None:
            if np.shape(hitmaps)[0] != nbands:
                raise AssertionError("Each band needs a hitmaps")

        if white_noises is not None:
            self.white_noises = np.array(white_noises)
            if np.shape(white_noises)[0] != nbands:
                raise AssertionError("Each band needs a whitenoise level")
        if ivar_map_filenames is not None:
            if np.shape(ivar_map_filenames)[0] != nbands:
                raise AssertionError("Each band needs a ivar_map_filenames")
        elif ivar_maps is not None:
            if np.shape(ivar_maps)[0] != nbands:
                raise AssertionError("Each band needs a ivar_map")

    def get_beams(self):
        return self.fwhms

    def get_hitmaps(self):
        return self.hitmaps

    def get_hitmap_filenames(self):
        return self.hitmap_filenames

    def get_ivar_maps(self):
        return self.ivar_maps

    def get_ivar_map_filenames(self):
        return self.ivar_map_filenames

    def get_white_noise(self, f_sky, units="arcmin2", *args, **kwargs):
        if self.white_noises is None:
            return None
        A = self.white_noises**2
        if units == "arcmin2":
            A *= (60 * 180 / np.pi) ** 2
        elif units != "sr":
            raise ValueError("Unknown units '%s'." % units)
        return A

    def get_noise_curves(
        self,
        f_sky,
        ell_max,
        delta_ell,
        deconv_beam=True,
        full_covar=False,
        rolloff_ell=None,
    ):
        ell = np.arange(2, ell_max, delta_ell)
        T_out = np.zeros([self.nbands, self.nbands, ell.shape[0]])
        P_out = np.zeros([self.nbands, self.nbands, ell.shape[0]])
        if self.noise_ell is not None:
            if len(np.shape(self.noise_TT)) == 3:
                for i in range(self.nbands):
                    for j in range(self.nbands):
                        # noise_ells,noise_T,noise_P = np.loadtxt(self.noise_files[i][j],unpack=True)
                        T_out[i, j] = interp1d(
                            self.noise_ell,
                            self.noise_TT[i, j],
                            bounds_error=False,
                            fill_value=0,
                        )(ell)
                        P_out[i, j] = interp1d(
                            self.noise_ell,
                            self.noise_PP[i, j],
                            bounds_error=False,
                            fill_value=0,
                        )(ell)
            else:
                for i in range(self.nbands):
                    T_out[i, i] = interp1d(
                        self.noise_ell,
                        self.noise_TT[i],
                        bounds_error=False,
                        fill_value=0,
                    )(ell)
                    P_out[i, i] = interp1d(
                        self.noise_ell,
                        self.noise_PP[i],
                        bounds_error=False,
                        fill_value=0,
                    )(ell)
        else:
            if np.shape(self.noise_files) == 2:
                for i in range(self.nbands):
                    for j in range(self.nbands):
                        noise_ell, noise_TT, noise_PP = np.loadtxt(
                            self.noise_files[i][j], unpack=True
                        )
                        T_out[i, j] = interp1d(
                            noise_ell, noise_TT, bounds_error=False, fill_value=0
                        )(ell)
                        P_out[i, j] = interp1d(
                            noise_ell, noise_TP, bounds_error=False, fill_value=0
                        )(ell)

            else:
                for i in range(self.nbands):
                    noise_ell, noise_TT, noise_PP = np.loadtxt(
                        self.noise_files[i], unpack=True
                    )
                    T_out[i, i] = interp1d(
                        noise_ell, noise_TT, bounds_error=False, fill_value=0
                    )(ell)
                    P_out[i, i] = interp1d(
                        noise_ell, noise_PP, bounds_error=False, fill_value=0
                    )(ell)
        return (ell, T_out, P_out)


if __name__ == "__main__":

    # Test that this code and the SONoiseSimulator agree if the above used the same noise curves and hitmap.
    nside = 16
    noise_sim_so = mapsims.noise.SONoiseSimulator(nside=nside)

    ell, noise_ell_T, noise_ell_P = noise_sim_so.get_survey("LT3").get_noise_curves(
        1.0,
        noise_sim_so.ell_max,
        delta_ell=1,
        full_covar=True,
        deconv_beam=noise_sim_so.apply_beam_correction,
        rolloff_ell=noise_sim_so.rolloff_ell,
    )

    hitsMaps, sky_fractions = noise_sim_so.get_hitmaps("LT3")

    noise_indices = noise_sim_so.get_noise_indices("LT3")
    hitsMaps_all = np.zeros([6, 12 * nside**2])
    hitsMaps_all[noise_indices] = hitsMaps
    soNoiseSim = noise_sim_so.simulate("LT3", seed=1)

    externSurvey = SurveyFromExternalData(
        6,
        fwhms=np.array([1.0] * 6) * u.arcmin,
        noise_ell=ell,
        noise_TT=noise_ell_T,
        noise_PP=noise_ell_P,
        hitmaps=hitsMaps_all,
    )
    chs = noise_sim_so.tubes["LT3"]

    noise_sim_extern = mapsims.noise.ExternalNoiseSimulator(
        nside=16, channels_list=chs, survey=externSurvey
    )

    simMaps = noise_sim_extern.simulate("LT3", seed=1)
    assert np.all(np.isclose(simMaps / soNoiseSim, 1))

    # A simple demonstration how to use if you wish to construct your own channel:
    test_ch = mapsims.channel_utils.Channel(
        tag="AA",
        telescope="AA",
        band="150",
        tube="PA1",
        beam=1.5 * u.arcmin,
        center_frequency=150 * u.GHz,
        noise_band_index=0,
        tube_id=10,
    )

    # For demonstrative purposes just use the noise curves above
    externSurvey = SurveyFromExternalData(
        1,
        fwhms=np.array([1.0]) * u.arcmin,
        noise_ell=ell,
        noise_TT=noise_ell_T[2:3, 2:3],
        noise_PP=noise_ell_P[2:3, 2:3],
        hitmaps=hitsMaps[0:1],
    )
    chs = [test_ch]

    noise_sim_extern = mapsims.noise.ExternalNoiseSimulator(
        nside=16, channels_list=chs, survey=externSurvey
    )

    simMaps_example = noise_sim_extern.simulate("PA1", seed=1)

    # A second example. Instead pass noise curves that tend to 1 at high ell and a inverse variance map

    whiteNoise = noise_sim_so.get_white_noise_power("LT3", 1)
    ivarMap = noise_sim_so.get_inverse_variance("LT3")

    # Scale the normal noise curves by the white noise levels
    # and pass a ivar map instead of hitmap.
    externSurvey_ivar = SurveyFromExternalData(
        1,
        fwhms=np.array([1.0]) * u.arcmin,
        noise_ell=ell,
        noise_TT=noise_ell_T[2:3, 2:3] / whiteNoise[0],
        noise_PP=noise_ell_P[2:3, 2:3] / whiteNoise[0],
        ivar_maps=ivarMap[:1],
    )

    noise_sim_extern_ivar = mapsims.noise.ExternalNoiseSimulator(
        nside=16, channels_list=chs, survey=externSurvey_ivar
    )

    simMaps_example_ivar = noise_sim_extern_ivar.simulate("PA1", seed=1)

    assert np.all(np.isclose(simMaps_example_ivar / simMaps_example, 1))
