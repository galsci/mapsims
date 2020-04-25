import os.path
import numpy as np
import healpy as hp
from astropy.utils import data
import warnings

import pysm.units as u

from so_models_v3 import SO_Noise_Calculator_Public_v3_1_1 as so_models
from . import so_utils
from . import utils as mutils

sensitivity_modes = {"baseline": 1, "goal": 2}
one_over_f_modes = {"pessimistic": 0, "optimistic": 1}
telescope_seed_offset = {"LA": 0, "SA": 1000}
default_mask_value = {"healpix": hp.UNSEEN, "car": np.nan}
_hitmap_version = "v0.2"

def band_ids_from_tube(tube):
    tubes = so_utils.tubes
    bands = tubes[tube]
    freqs = return [so_utils.band_freqs[x] for x in bands]
    available_frequencies = np.unique(so_utils.frequencies)
    band_ids = [available_frequencies.searchsorted(f) for f in freqs]
    return band_ids


class SONoiseSimulator:
    def __init__(
        self,
        telescopes=["LA"],
        nside=None,
        shape=None,
        wcs=None,
        ell_max=None,
        num=None,
        return_uK_CMB=True,
        sensitivity_mode="baseline",
        apply_beam_correction=True,
        apply_kludge_correction=True,
        homogenous=False,
        no_power_below_ell=None,
        rolloff_ell=None,
        survey_efficiency=0.2,
        full_covariance=False,
        LA_years=5,
        LA_noise_model="SOLatV3point1",
        elevation=50,
        SA_years=5,
        SA_one_over_f_mode="pessimistic",
        sky_fraction = None,
        cache_hitmaps = True,
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
        num : int
            Numpy random seed, each band is going to get a different seed as seed + band + (1000 for SA)
        return_uK_CMB : bool
            True, output is in microK_CMB, False output is in microK_RJ
        sensitivity_mode : str
            Value should be threshold, baseline or goal to use predefined sensitivities
        apply_beam_correction : bool
            Include the effect of the beam in the noise angular power spectrum
        apply_kludge_correction : bool
            If True, reduce the hitcount by a factor of 0.85 to account for not-uniformity in the scanning
        homogenous : bool
            Set to True to generate full-sky maps with no hit-count variation, with noise curves
            corresponding to a survey that covers a sky fraction of sky_fraction (defaults to 1).
        no_power_below_ell : int
            The input spectra have significant power at low ell, we can zero that power specifying an integer
            :math:`\ell` value here. The power spectra at :math:`\ell < \ell_0` are set to zero.
        rolloff_ell : int
            Low ell power damping, see the docstring of so_noise_models.so_models_v3.SO_Noise_Calculator_Public_v3_1_1.rolloff
        survey_efficiency : float
            Fraction of calendar time that may be used to compute map depth.
        full_covariance : bool
        LA_years : int
            Total number of years for the Large Aperture telescopes survey
        LA_number_LF : int
            Number of Low Frequency tubes in LAT
        LA_number_MF : int
            Number of Medium Frequency tubes in LAT
        LA_number_UHF : int
            Number of Ultra High Frequency tubes in LAT
        LA_noise_model : str
            Noise model among the ones available in `so_noise_model`, "SOLatV3point1" is default, "SOLatV3" is
            the model released in 2018 which had a bug in the atmosphere contribution
        elevation : float
            Elevation of the scans in degrees, the V3.1.1 noise model includes elevation
            dependence for the LAT. This should reproduced original V3 results at the
            reference elevation of 50 degrees.
        SA_years : int
            Total number of years for the Small Aperture telescopes survey
        SA_number_*: survey-averaged number of each SAT tube in operation.
            For example, the default is 0.4 LF, 1.6 MF, and 1 UHF]
            populating a total of 3 tubes.  Fractional tubes are acceptable
            (imagine a tube were swapped out part way through the
            survey).
        SA_one_over_f_mode : {"pessimistic", "optimistic", "none"}
            Correlated noise performance of the detectors on the Small Aperture telescopes
        sky_fraction : optional,float
            If homogenous is True, this sky_fraction is used for the noise curves.
        cache_hitmaps : bool
            If True, caches hitmaps.
        boolean_sky_fraction: bool
            If True, determines fsky based on fraction of hitmap that is zero. If False,
            determines sky_fraction from <Nhits>.
        """

        if nside is None:
            assert shape is not None
            assert wcs is not None
            self.healpix = False
            self.shape = shape[-2:]
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

        self.rolloff_ell = rolloff_ell
        self._sky_fraction = sky_fraction
        self.sensitivity_mode = sensitivity_modes[sensitivity_mode]
        self.apply_beam_correction = apply_beam_correction
        self.apply_kludge_correction = apply_kludge_correction
        self.survey_efficiency = survey_efficiency
        if self.apply_kludge_correction:
            self.survey_efficiency *= 0.85
        self.full_covariance = full_covariance
        self.seed = num
        self.return_uK_CMB = return_uK_CMB
        self.no_power_below_ell = no_power_below_ell
        self.LA_years = LA_years
        self.LA_number_LF = LA_number_LF
        self.LA_number_MF = LA_number_MF
        self.LA_number_UHF = LA_number_UHF
        self.LA_noise_model = LA_noise_model
        self.elevation = elevation
        self.SA_years = SA_years
        self.SA_number_LF = SA_number_LF
        self.SA_number_MF = SA_number_MF
        self.SA_number_UHF = SA_number_UHF
        self.SA_one_over_f_mode = one_over_f_modes[SA_one_over_f_mode]
        self.homogenous = homogenous

        self.hitmap_version = _hitmap_version
        self._cache = cache_hitmaps
        if self._cache: self._hmap_cache = {}
        self.remote_data = mutils.RemoteData(
            healpix=self.healpix, version=self.hitmap_version
        )


    def load_noise_spectra(self, tube, ncurve_fsky=1):
        """Update a telescope configuration by loading the corresponding
        hitmaps. Each loaded `telescope` is kept in memory, but
        new choice of `scanning_strategy` erases the previous one.

        Parameters
        ----------

        telescope : string
            Telescope identifier, typically `LA` or `SA` for the large aperture
            and small aperture, respectively.

        """

        telescope = f'{tube[0]}A' # get LA or SA from tube name

        if telescope == "SA":
            if   tube=='ST0': N_tubes = [0,0,1] # the UHF telescope
            elif tube=='ST1': N_tubes = [0,1,0] # the MF telescope
            elif tube=='ST2': N_tubes = [0,0.6,0] # the MF/LF telescope
            elif tube=='ST3': N_tubes = [0.4,0,0] # the MF/LF telescope
            else: raise ValueError

            survey = so_models.SOSatV3point1(
                sensitivity_mode=self.sensitivity_mode,
                survey_efficiency=self.survey_efficiency,
                survey_years=self.SA_years,
                N_tubes=N_tubes,
                el=None,  # SAT does not support noise elevation function
                one_over_f_mode=self.SA_one_over_f_mode,
            )
            ell, noise_ell_T, noise_ell_P = survey.get_noise_curves(
                ncurve_fsky,  # We load hitmaps later, so we compute and apply sky fraction later
                self.ell_max,
                delta_ell=1,
                full_covar=True,
                deconv_beam=self.apply_beam_correction,
                rolloff_ell=self.rolloff_ell,
            )
            # For SA, so_noise simulates only Polarization,
            # Assume that T is half
            if noise_ell_T is None:
                noise_ell_T = noise_ell_P / 2
        elif telescope == "LA":
            survey = getattr(so_models, self.LA_noise_model)(
                sensitivity_mode=self.sensitivity_mode,
                survey_efficiency=self.survey_efficiency,
                survey_years=self.LA_years,
                N_tubes=[1,1,1],
                el=self.elevation,
            )
            ell, noise_ell_T, noise_ell_P = survey.get_noise_curves(
                ncurve_fsky,  # We load hitmaps later, so we compute and apply sky fraction later
                self.ell_max,
                delta_ell=1,
                full_covar=True,
                deconv_beam=self.apply_beam_correction,
                rolloff_ell=self.rolloff_ell,
            )

        self.ell = np.arange(ell[-1] + 1)
        output_noise_ell_T = {}
        output_noise_ell_P = {}

        output_noise_ell_T[frequency] = np.insert(
            noise_ell_T[band_index][band_index], 0, [0, 0]
        )
        output_noise_ell_P[frequency] = np.insert(
            noise_ell_P[band_index][band_index], 0, [0, 0]
        )
        if frequency in so_utils.frequencies_with_correlations:
            output_noise_ell_T[str(frequency) + "_corr"] = np.insert(
                noise_ell_T[band_index][band_index + 1], 0, [0, 0]
            )
            output_noise_ell_P[str(frequency) + "_corr"] = np.insert(
                noise_ell_P[band_index][band_index + 1], 0, [0, 0]
            )

        if self.no_power_below_ell is not None:
            output_noise_ell_T[frequency][self.ell < self.no_power_below_ell] = 0
            output_noise_ell_P[frequency][self.ell < self.no_power_below_ell] = 0
        return output_noise_ell_T, output_noise_ell_P

    def _validate_map(self,fmap):
        shape = fmap.shape
        if self.healpix:
            if len(shape)==1: 

    def _load_map(self,fname,**kwargs):
        # If not a string try as a healpix or CAR map
        if not(isinstance(fname,string)): return _validate_map(fname)
        if self.healpix: return hp.read_map(fname,**kwargs)
        else: 
            from pixell import enmap
            return enmap.read_map(fname,**kwargs)

    def _process_hitmap(self,hmap):
        if self.boolean_sky_fraction:
            pass
        else:
            
            

    def load_hitmap(self, tube=None, hitmap=None):

        if hitmap is not None: return _process_hitmap(_load_map(hitmap))

        telescope = f'{tube[0]}A' # get LA or SA from tube name

        if not (self.healpix):
            npixheight = min(
                {"LA": [0.5, 2.0], "SA": [4.0, 12.0]}[telescope],
                key=lambda x: abs(x - self._pixheight),
            )
            car_suffix = f"_CAR_{npixheight:.2f}_arcmin"
        else:
            car_suffix = ""

        rnames = []
        for ch in chs:
            rnames.append(
                f"{tube}_{ch.band}_01_of_20.nominal_telescope_all_time_all_hmap{car_suffix}.fits.gz"
            )

        hitmap_filenames = [self.remote_data.get(rname) for rname in rnames]

        if self.healpix:
            hitmaps = [
                hp.ud_grade(
                    hp.read_map(hitmap_filename, verbose=False), nside_out=self.nside
                )
                for hitmap_filename in hitmap_filenames
            ]
        else:
            from pixell import enmap, wcsutils

            hitmaps = [
                enmap.read_map(hitmap_filename) for hitmap_filename in hitmap_filenames
            ]
            if wcsutils.is_compatible(hitmaps[0].wcs, self.wcs):
                hitmaps = [
                    enmap.extract(hitmap, self.shape, self.wcs) for hitmap in hitmaps
                ]
            else:
                warnings.warn(
                    "WCS of hitmap with nearest pixel-size is not compatible, so interpolating hitmap"
                )
                hitmaps = [
                    enmap.project(hitmap, self.shape, self.wcs,order=0) for hitmap in hitmaps
                ]

        for hitmap in hitmaps:
            hitmap /= hitmap.max()
        # Discard pixels with very few hits that cause border effects
        # hitmap[hitmap < 1e-3] = 0
        if self.healpix:
            sky_fractions = [(hitmap != 0).sum() / hitmap.size for hitmap in hitmaps]
        else:
            pmap = enmap.pixsizemap(self.shape, self.wcs)
            sky_fractions = [(
                pmap[hitmap != 0].sum()
                / 4.0
                / np.pi
            ) for hitmap in hitmaps]

        if len(hitmaps) == 1:
            hitmaps = hitmaps[0]
            sky_fractions = sky_fractions[0]
        return hitmaps, sky_fractions

    def get_white_noise_power(self, ch, units='sr'):
        """Get white noise power in uK^2-sr (units='sr') or
        uK^2-arcmin^2 (units='arcmin2') corresponding to the channel identifier ch.
        This is useful if you want to generate your own simulations that do not
        have the atmospheric component.

        Parameters
        ----------

        ch : mapsims.Channel
            Channel identifier, create with e.g. mapsims.SOChannel("SA", 27)

        """
        available_frequencies = np.unique(so_utils.frequencies)
        frequency = ch.center_frequency.value
        band_index = available_frequencies.searchsorted(frequency)
        f_sky = self.sky_fraction[ch.telescope]
        return self.surveys[ch.telescope].get_white_noise(f_sky, units=units)[band_index]


    def simulate(
        self,
        tube,
        output_units="uK_CMB",
        seed=None,
        nsplits=1,
        mask_value=None,
        atmosphere=True,
        hitmap=None,
            
    ):
        """Create a random realization of the noise power spectrum

        Parameters
        ----------

        ch : mapsims.Channel
            Channel identifier, create with e.g. mapsims.SOChannel("SA", 27)
            Optional, we can specify a tube and simulate both channels
        tube : str
            Specify a specific tube, required for hitmaps v0.2, for available
            tubes and their channels, see so_utils.tubes.
        output_units : str
            Output unit supported by PySM.units, e.g. uK_CMB or K_RJ
        seed : integer, optional
            Specify a seed, if not specified, we use self.seed and then offset it
            differently for each channel.
        nsplits : integer, optional
            Number of splits to generate. The splits will have independent noise
            realizations, with noise power scaled by a factor of nsplits, i.e. atmospheric
            noise is assumed to average down with observing time the same way
            the white noise does. By default, only one split (the coadd) is generated.
        mask_value : float, optional
            The value to set in masked (unobserved) regions. By default, it uses
            the value in default_mask_value, which for healpix is healpy.UNSEEN
            and for CAR is numpy.nan.
        atmosphere : bool, optional
            Whether to include the correlated 1/f from the noise model. This is
            True by default. If it is set to False, then a pure white noise map
            is generated from the white noise power in the noise model.
        hitmap


        Returns
        -------

        output_map : ndarray
            Numpy array with the HEALPix map realization of noise
        """
        assert nsplits >= 1
        if mask_value is None:
            mask_value = (
                default_mask_value["healpix"]
                if self.healpix
                else default_mask_value["car"]
            )


        if seed is not None:
            np.random.seed(seed)
        else:
            if self.seed is not None:
                try:
                    frequency_offset = int(chs[0].band)
                except ValueError:
                    frequency_offset = so_utils.bands.index(chs[0].band) * 100
                np.random.seed(
                    self.seed
                    + frequency_offset
                    + telescope_seed_offset[chs[0].telescope]
                    + (
                        nsplits - 1
                    )  # avoid any risk of same seed between full and the first split
                )

        if self.scanning_strategy is False:
            ones = np.ones(hp.nside2npix(self.nside)) if self.healpix else enmap.ones(self.shape,self.wcs)
            hitmaps = [ones, ones] if self.full_covariance else ones
            fsky = self._sky_fraction if self._sky_fraction is not None else 1
            sky_fractions = [fsky, fsky] if self.full_covariance else fsky
        else:
            hitmaps, sky_fractions = self.load_hitmaps(chs, tube,hitmap=hitmap)

        ps_T = (
            np.asarray(
                [
                    self.noise_ell_T[chs[0].telescope][
                        int(chs[0].center_frequency.value)
                    ]
                    * sky_fractions[0],
                    self.noise_ell_T[chs[1].telescope][
                        int(chs[1].center_frequency.value)
                    ]
                    * sky_fractions[1],
                    self.noise_ell_T[chs[0].telescope][
                        str(int(chs[0].center_frequency.value)) + "_corr"
                    ]
                    * np.mean(sky_fractions),
                ]
            )
            * nsplits
        )

        ps_P = (
            np.asarray(
                [
                    self.noise_ell_P[chs[0].telescope][
                        int(chs[0].center_frequency.value)
                    ]
                    * sky_fractions[0],
                    self.noise_ell_P[chs[1].telescope][
                        int(chs[1].center_frequency.value)
                    ]
                    * sky_fractions[1],
                    self.noise_ell_P[chs[0].telescope][
                        str(int(chs[0].center_frequency.value)) + "_corr"
                    ]
                    * np.mean(sky_fractions),
                ]
            )
            * nsplits
        )

        if not(self.full_covariance):
            ps_T[2] = 0
            ps_P[2] = 0


        if self.healpix:
            npix = hp.nside2npix(self.nside)
            output_map = np.zeros((2, nsplits, 3, npix))
            for i in range(nsplits):
                for i_pol in range(3):
                    output_map[0][i][i_pol], output_map[1][i][i_pol] = np.array(
                        hp.synfast(
                            ps_T if i_pol == 0 else ps_P,
                            nside=self.nside,
                            pol=False,
                            new=True,
                            verbose=False,
                        )
                    )
        else:
            from pixell import curvedsky, powspec
            output_map = np.zeros((2, nsplits, 3, ) + self.shape)
            ps_T = powspec.sym_expand(np.asarray(ps_T), scheme="diag")
            ps_P = powspec.sym_expand(np.asarray(ps_P), scheme="diag")
            # TODO: These loops can probably be vectorized
            for i in range(nsplits):
                for i_pol in range(3):
                    output_map[:,i,i_pol] = curvedsky.rand_map((2,) + self.shape, self.wcs, ps_T if i_pol==0 else ps_P)


        for out_map, hitmap, sky_fraction, ch in zip(
            output_map, hitmaps, sky_fractions, chs
        ):

            good = hitmap != 0
            # Normalize on the Effective sky fraction, see discussion in:
            # https://github.com/simonsobs/mapsims/pull/5#discussion_r244939311
            out_map[:, :, good] /= np.sqrt(hitmap[good] / hitmap.mean() * sky_fraction)
            out_map[:, :, np.logical_not(good)] = mask_value
            unit_conv = (1 * u.uK_CMB).to_value(
                u.Unit(output_units),
                equivalencies=u.cmb_equivalencies(ch.center_frequency),
            )
            out_map *= unit_conv

        if not self.full_covariance:
            output_map = output_map[0]
        if nsplits == 1:
            if self.full_covariance:
                return output_map[:, 0, :, :]
            else:
                return output_map[0]

        return output_map
