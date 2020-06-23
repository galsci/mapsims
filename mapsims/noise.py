from collections import defaultdict
import numpy as np
import healpy as hp
import warnings

try:  # PySM >= 3.2.1
    import pysm3.units as u
except ImportError:
    import pysm.units as u

from so_models_v3 import SO_Noise_Calculator_Public_v3_1_1 as so_models

# pixell is optional and needed when CAR simulations are requested
try:
    import pixell
    import pixell.curvedsky
    import pixell.powspec
except:
    pixell = None

from .channel_utils import parse_channels
from .utils import DEFAULT_INSTRUMENT_PARAMETERS, RemoteData

sensitivity_modes = {"baseline": 1, "goal": 2}
one_over_f_modes = {"pessimistic": 0, "optimistic": 1}
default_mask_value = {"healpix": hp.UNSEEN, "car": np.nan}
_hitmap_version = "v0.2"


class SONoiseSimulator:
    def __init__(
        self,
        nside=None,
        shape=None,
        wcs=None,
        ell_max=None,
        return_uK_CMB=True,
        sensitivity_mode="baseline",
        apply_beam_correction=False,
        apply_kludge_correction=True,
        homogeneous=False,
        no_power_below_ell=None,
        rolloff_ell=50,
        survey_efficiency=0.2,
        full_covariance=True,
        LA_years=5,
        LA_noise_model="SOLatV3point1",
        elevation=50,
        SA_years=5,
        SA_one_over_f_mode="pessimistic",
        sky_fraction=None,
        cache_hitmaps=True,
        boolean_sky_fraction=False,
        instrument_parameters=DEFAULT_INSTRUMENT_PARAMETERS,
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
            nside of HEALPix map. If None, uses
            rectangular pixel geometry specified through shape and wcs.
        shape : tuple of ints
            shape of ndmap array (see pixell.pixell.enmap). Must also specify wcs.
        wcs : astropy.wcs.wcs.WCS instance
            World Coordinate System for geometry of map (see pixell.pixell.enmap). Must
            also specify shape.
        ell_max : int
            Maximum ell for the angular power spectrum, if not provided set to 3 * nside when using healpix
            or 10000 * (1.0 / pixel_height_arcmin) when using CAR, corresponding roughly to the Nyquist
            frequency.
        return_uK_CMB : bool
            True, output is in microK_CMB, False output is in microK_RJ
        sensitivity_mode : str
            Value should be threshold, baseline or goal to use predefined sensitivities
        apply_beam_correction : bool
            Include the effect of the beam in the noise angular power spectrum
        apply_kludge_correction : bool
            If True, reduce the hitcount by a factor of 0.85 to account for not-uniformity in the scanning
        homogeneous : bool
            Set to True to generate full-sky maps with no hit-count variation, with noise curves
            corresponding to a survey that covers a sky fraction of sky_fraction (defaults to 1).
        no_power_below_ell : int
            The input spectra have significant power at low :math:`\ell`,
            we can zero that power specifying an integer :math:`\ell` value here.
            The power spectra at :math:`\ell < \ell_0` are set to zero.
        rolloff_ell : int
            Low ell power damping, see the docstring of
            `so_noise_models.so_models_v3.SO_Noise_Calculator_Public_v3_1_1.rolloff`
        survey_efficiency : float
            Fraction of calendar time that may be used to compute map depth.
        full_covariance : bool
            Whether or not to include the intra-tube covariance between bands.
            If white noise (atmosphere=False) sims are requested, no
            covariance is included regardless of the value of full_covariance.
        LA_years : int
            Total number of years for the Large Aperture telescopes survey
        LA_noise_model : str
            Noise model among the ones available in `so_noise_model`, "SOLatV3point1" is default, "SOLatV3" is
            the model released in 2018 which had a bug in the atmosphere contribution
        elevation : float
            Elevation of the scans in degrees, the V3.1.1 noise model includes elevation
            dependence for the LAT. This should reproduced original V3 results at the
            reference elevation of 50 degrees.
        SA_years : int
            Total number of years for the Small Aperture telescopes survey
        SA_one_over_f_mode : {"pessimistic", "optimistic", "none"}
            Correlated noise performance of the detectors on the Small Aperture telescopes
        sky_fraction : optional,float
            If homogeneous is True, this sky_fraction is used for the noise curves.
        cache_hitmaps : bool
            If True, caches hitmaps.
        boolean_sky_fraction: bool
            If True, determines sky fraction based on fraction of hitmap that is zero. If False,
            determines sky_fraction from <Nhits>.
        instrument_parameters : Path or str
            See the help of MapSims
        """

        channels_list = parse_channels(instrument_parameters=instrument_parameters)
        self.channels = {ch.tag: ch for ch in channels_list}
        self.tubes = defaultdict(list)
        for ch in channels_list:
            self.tubes[ch.tube].append(ch)

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
            self.pixarea_map = pixell.enmap.pixsizemap(self.shape, self.wcs)
            self.map_area = pixell.enmap.area(self.shape, self.wcs)
        else:
            assert shape is None
            assert wcs is None
            self.healpix = True
            self.nside = nside
            self.ell_max = ell_max if ell_max is not None else 3 * nside
            self.pixarea_map = hp.nside2pixarea(nside)
            self.map_area = 4.0 * np.pi

        self.rolloff_ell = rolloff_ell
        self.boolean_sky_fraction = boolean_sky_fraction
        self._sky_fraction = sky_fraction
        self.sensitivity_mode = sensitivity_modes[sensitivity_mode]
        self.apply_beam_correction = apply_beam_correction
        self.apply_kludge_correction = apply_kludge_correction
        self.survey_efficiency = survey_efficiency
        if self.apply_kludge_correction:
            self.survey_efficiency *= 0.85
        self.full_covariance = full_covariance
        self.return_uK_CMB = return_uK_CMB
        self.no_power_below_ell = no_power_below_ell
        self.LA_years = LA_years
        self.LA_noise_model = LA_noise_model
        self.elevation = elevation
        self.SA_years = SA_years
        self.SA_one_over_f_mode = one_over_f_modes[SA_one_over_f_mode]
        self.homogeneous = homogeneous

        self.hitmap_version = _hitmap_version
        self._cache = cache_hitmaps
        self._hmap_cache = {}
        self.remote_data = RemoteData(healpix=self.healpix, version=self.hitmap_version)

    def get_beam_fwhm(self, tube, band=None):
        """Get beam FWHMs in arcminutes corresponding to the tueb.
        This is useful if non-beam-deconvolved sims are requested and you want to
        know what beam to apply to your signal simulation.

        Parameters
        ----------

        tube : str
            Specify a tube (for SO: ST0-ST3, LT0-LT6) see the `tubes` attribute

        band : str,optional
            Optionally specify the band name within the tube to get just its
            white noise.


        Returns
        -------

        beam : tuple of floats
            The beam FWHM in arcminutes either as
            a tuple for the pair of bands in the tube, or just for the specific
            band requested.
            See the `band_id` attribute of the Channel class
            to identify which is the index of a Channel in the array.
        """

        survey = self._get_survey(tube)
        noise_indices = self.get_noise_indices(tube, band)
        return survey.get_beams()[noise_indices]

    def get_noise_indices(self, tube, band=None):
        """Gets indices in the so_noise_model package of a channel or the 2 channels of a tube
        """
        if band is None:
            band_indices = [ch.noise_band_index for ch in self.tubes[tube]]
        else:
            band_indices = self.channels[tube + "_" + band].noise_band_index
        return band_indices

    def _get_survey(self, tube):
        """Internal function to get the survey object
        from the SO noise model code.
        """
        telescope = f"{tube[0]}A"  # get LA or SA from tube name
        if telescope == "SA":
            if tube == "ST0":
                N_tubes = [0, 0, 1]  # the UHF telescope
            elif tube == "ST1":
                N_tubes = [0, 1, 0]  # the MF telescope
            elif tube == "ST2":
                N_tubes = [0, 0.6, 0]  # the MF/LF telescope
            elif tube == "ST3":
                N_tubes = [0.4, 0, 0]  # the MF/LF telescope
            else:
                raise ValueError

            with np.errstate(divide="ignore", invalid="ignore"):
                survey = so_models.SOSatV3point1(
                    sensitivity_mode=self.sensitivity_mode,
                    survey_efficiency=self.survey_efficiency,
                    survey_years=self.SA_years,
                    N_tubes=N_tubes,
                    el=None,  # SAT does not support noise elevation function
                    one_over_f_mode=self.SA_one_over_f_mode,
                )
        elif telescope == "LA":
            with np.errstate(divide="ignore", invalid="ignore"):
                survey = getattr(so_models, self.LA_noise_model)(
                    sensitivity_mode=self.sensitivity_mode,
                    survey_efficiency=self.survey_efficiency,
                    survey_years=self.LA_years,
                    N_tubes=[1, 1, 1],
                    el=self.elevation,
                )
        return survey

    def get_fullsky_noise_spectra(self, tube, ncurve_sky_fraction=1, return_corr=False):
        """Get the noise power spectra corresponding to the requested tube
        from the SO noise model code.

        See get_noise_properties to get spectra scaled with the proper hitmap

        See the `band_id` attribute of the Channel class
        to identify which is the index of a Channel in the returned arrays.

        Parameters
        ----------

        tube : str
            Specify a tube (for SO: ST0-ST3, LT0-LT6) see the `tubes` attribute

        ncurve_sky_fraction : float,optional
            The sky fraction to report to the noise simulator code.
            In the current implementation, the default is to pass
            a sky fraction of 1, and scale the result by
            the corresponding sky fraction determined from each
            band's hitmap.

        return_corr : bool
            If True, returns cross-correlation N_XY / sqrt(N_XX * N_YY) coeffient instead of
            cross-correlation power N_XY in the third row of the returned arrays. This is
            more convenient sometimes, e.g. when you need to scale the auto-correlation power by some factor.

        Returns
        -------

        ell : (nells,) ndarray
            Array of nells multipoles starting at ell=0 and spaced by delta_ell=1
            corresponding to the noise power spectra nells_T and nells_P

        nells_T : (3,nells) ndarray
            The first two rows contain the temperature auto-correlation of the noise power
            spectra of each band in the tube. The third row contains the correlation
            power between the two bands by default, but you can get
            the cross-correlation coefficient instead by setting return_corr=True.

        nells_P : (3,nells) ndarray
            Same as for nells_T but for polarization.

        """

        telescope = f"{tube[0]}A"  # get LA or SA from tube name
        survey = self._get_survey(tube)
        if telescope == "SA":
            ell, noise_ell_T, noise_ell_P = survey.get_noise_curves(
                ncurve_sky_fraction,
                self.ell_max,
                delta_ell=1,
                full_covar=True,  # we always obtain the full covariance and later remove correlations as necessary
                deconv_beam=self.apply_beam_correction,
                rolloff_ell=self.rolloff_ell,
            )
            # For SA, so_noise simulates only Polarization,
            # Assume that T is half
            if noise_ell_T is None:
                noise_ell_T = noise_ell_P / 2
        elif telescope == "LA":
            ell, noise_ell_T, noise_ell_P = survey.get_noise_curves(
                ncurve_sky_fraction,
                self.ell_max,
                delta_ell=1,
                full_covar=True,
                deconv_beam=self.apply_beam_correction,
                rolloff_ell=self.rolloff_ell,
            )

        assert ell[0] == 2  # make sure the noise code is still returning something
        # that starts at ell=2
        ls = np.arange(ell.size + 2)
        nells_T = np.zeros((3, ell.size + 2))
        nells_P = np.zeros((3, ell.size + 2))
        b1, b2 = [ch.noise_band_index for ch in self.tubes[tube]]
        for n_out, n_in in zip([nells_T, nells_P], [noise_ell_T, noise_ell_P]):
            n_out[0, 2:] = n_in[b1][b1]
            n_out[1, 2:] = n_in[b2][b2]
            # re-scaling if correlation coefficient is requested
            scale = np.sqrt(n_in[b1][b1] * n_in[b2][b2]) if return_corr else 1
            n_out[2, 2:] = (n_in[b1][b2] / scale) if self.full_covariance else 0
            if self.no_power_below_ell is not None:
                n_out[:, ls < self.no_power_below_ell] = 0

        ell = ls
        return ell, nells_T, nells_P

    def _validate_map(self, fmap):
        """Internal function to validate an externally provided map.
        It checks the healpix or CAR attributes against what the
        class was initialized with. It adds a leading dimension if
        necessary.
        """
        shape = fmap.shape
        if self.healpix:
            if len(shape) == 1:
                npix = shape[0]
            elif len(shape) == 2:
                assert shape[0] == 1 or shape[0] == 2
                npix = shape[1]
            else:
                raise ValueError
            assert npix == hp.nside2npix(self.nside)
            if len(shape) == 1:
                return fmap[None, :]
            else:
                return fmap
        else:
            assert pixell.wcsutils.is_compatible(fmap.wcs, self.wcs)
            if len(shape) == 2:
                ashape = shape
            elif len(shape) == 3:
                assert shape[0] == 1 or shape[0] == 2
                ashape = shape[-2:]
            else:
                raise ValueError
            assert [x == y for x, y in zip(ashape, self.shape)]
            if len(shape) == 2:
                return fmap[None, ...]
            else:
                return fmap

    def _load_map(self, fname, **kwargs):
        """Internal function to load a healpix or CAR map
        from disk and reproject it if necessary.
        """
        # If not a string try as a healpix or CAR map
        if not (isinstance(fname, str)):
            return self._validate_map(fname)

        # Check if in cache
        if self._cache:
            try:
                return self._hmap_cache[fname]
            except:
                pass

        # else load
        if self.healpix:
            hitmap = hp.ud_grade(
                hp.read_map(fname, verbose=False), nside_out=self.nside
            )
        else:

            hitmap = pixell.enmap.read_map(fname)
            if pixell.wcsutils.is_compatible(hitmap.wcs, self.wcs):
                hitmap = pixell.enmap.extract(hitmap, self.shape, self.wcs)
            else:
                warnings.warn(
                    "WCS of hitmap with nearest pixel-size is not compatible, so interpolating hitmap"
                )
                hitmap = pixell.enmap.project(hitmap, self.shape, self.wcs, order=0)

        # and then cache and return
        if self._cache:
            self._hmap_cache[fname] = hitmap
        return hitmap

    def _average(self, imap):
        # Internal function to calculate <imap> general to healpix and CAR
        if self.healpix:
            assert imap.ndim == 1
        else:
            assert imap.ndim == 2
        return (self.pixarea_map * imap).sum() / (self.map_area)

    def _process_hitmaps(self, hitmaps):
        """Internal function to process hitmaps and based on the
        desired scheme, obtain sky fractions from them.
        """
        nhitmaps = hitmaps.shape[0]
        assert nhitmaps == 1 or nhitmaps == 2
        if self.boolean_sky_fraction:
            raise NotImplementedError
        else:
            for i in range(nhitmaps):
                hitmaps[i] /= hitmaps[i].max()

            # We define sky fraction as <Nhits>
            sky_fractions = [self._average(hitmaps[i]) for i in range(nhitmaps)]
        return hitmaps, sky_fractions

    def get_hitmaps(self, tube=None, hitmap=None):
        """Get and process hitmaps and sky fractions for the provided tube or provided
        an external one.

        Parameters
        ----------

        tube : str
            Specify a tube (for SO: ST0-ST3, LT0-LT6) see the `tubes` attribute

        hitmap : string or map, optional
            Provide the path to a hitmap to override the default used for
            the tube. You could also provide the hitmap as an array
            directly.

        Returns
        -------

        hitmaps : ndarray or ndmap
            Processed hitmaps. See the `band_id` attribute of the Channel class
            to identify which is the index of a Channel in the array.

        sky_fractions : float
            The sky fraction covered by the survey determined from the hitmaps.

        """

        if hitmap is not None:
            return self._process_hitmaps(self._load_map(hitmap))

        telescope = f"{tube[0]}A"  # get LA or SA from tube name

        if not (self.healpix):
            npixheight = min(
                {"LA": [0.5, 2.0], "SA": [4.0, 12.0]}[telescope],
                key=lambda x: abs(x - self._pixheight),
            )
            car_suffix = f"_CAR_{npixheight:.2f}_arcmin"
        else:
            car_suffix = ""

        bands = [ch.band for ch in self.tubes[tube]]

        rnames = []
        for band in bands:
            rnames.append(
                f"{tube}_{band}_01_of_20.nominal_telescope_all_time_all_hmap{car_suffix}.fits.gz"
            )

        hitmap_filenames = [self.remote_data.get(rname) for rname in rnames]
        hitmaps = []
        for hitmap_filename in hitmap_filenames:
            hitmaps.append(self._load_map(hitmap_filename))

        return self._process_hitmaps(np.asarray(hitmaps))

    def get_white_noise_power(self, tube, sky_fraction, band=None, units="sr"):
        """Get white noise power in uK^2-sr (units='sr') or
        uK^2-arcmin^2 (units='arcmin2') corresponding to the tube name tube.
        This is useful if you want to generate your own simulations that do not
        have the atmospheric component.

        Parameters
        ----------

        tube : str
            Specify a tube (for SO: ST0-ST3, LT0-LT6) see the `tubes` attribute

        sky_fraction : float
            The sky fraction covered by the survey.

        band : str,optional
            Optionally specify the band name within the tube to get just its
            white noise.

        units: str
            'sr' for white noise power in uK^2-steradian and 'arcmin2' for
            the same in uK^2-arcmin^2 units.

        Returns
        -------

        wnoise : tuple of floats
            The white noise variance in the requested units either as
            a tuple for the pair of bands in the tube, or just for the specific
            band requested.
            See the `band_id` attribute of the Channel class
            to identify which is the index of a Channel in the array.
        """
        survey = self._get_survey(tube)
        noise_indices = self.get_noise_indices(tube, band)
        return survey.get_white_noise(sky_fraction, units=units)[noise_indices]

    def get_inverse_variance(
        self, tube, output_units="uK_CMB", hitmap=None, white_noise_rms=None
    ):
        """Get the inverse noise variance in each pixel for the requested tube.
        In the noise model, all the splits and all the I,Q,U components have the 
        same position dependence of the noise variance. Each split just has `nsplits` 
        times the noise power (or `1/nsplits` the inverse noise variance) and the 
        Q,U components have 2x times the noise power (or 1/2 times the inverse 
        noise variance) of the intensity components. The inverse noise variance 
        provided by this function is for the `nsplits=1` intensity component. 
        Two maps are stored in the leading dimension, one for each of the 
        two correlated arrays in the dichroic tube.


        Parameters
        ----------

        tube : str
            Specify a tube (for SO: ST0-ST3, LT0-LT6) see the `tubes` attribute
        output_units : str
            Output unit supported by PySM.units, e.g. uK_CMB or K_RJ
        hitmap : string or map, optional
            Provide the path to a hitmap to override the default used for
            the tube. You could also provide the hitmap as an array
            directly.
        white_noise_rms : float or tuple of floats, optional
            Optionally scale the simulation so that the small-scale limit white noise
            level is white_noise_rms in uK-arcmin (either a single number or
            a pair for the dichroic array).

        Returns
        -------

        ivar_map : ndarray or ndmap
            Numpy array with the HEALPix or CAR map of the inverse variance
            in each pixel. The default units are uK^(-2). This is an extensive
            quantity that depends on the size of pixels.
            See the `band_id` attribute of the Channel class
            to identify which is the index of a Channel in the array.
        """
        fsky, hitmaps = self._get_requested_hitmaps(tube, hitmap)
        wnoise_scale = self._get_wscale_factor(white_noise_rms, tube, fsky)
        sel = np.s_[:, None] if self.healpix else np.s_[:, None, None]
        power = (
            self.get_white_noise_power(tube, sky_fraction=1, units="sr")[sel]
            * fsky[sel]
            * wnoise_scale[:, 0][sel]
        )
        """
        We now have the physical white noise power uK^2-sr
        and the hitmap
        ivar = hitmap * pixel_area * fsky / <hitmap> / power
        """
        avgNhits = np.asarray([self._average(hitmaps[i]) for i in range(2)])
        ret = hitmaps * self.pixarea_map * fsky[sel] / avgNhits[sel] / power
        # Convert to desired units
        for i in range(2):
            freq = self.tubes[tube][i].center_frequency
            unit_conv = (1 * u.uK_CMB).to_value(
                u.Unit(output_units), equivalencies=u.cmb_equivalencies(freq)
            )
            ret[i] /= unit_conv ** 2.0  # divide by square since the default is 1/uK^2
        return ret

    def _get_wscale_factor(self, white_noise_rms, tube, sky_fraction):
        """Internal function to re-scale white noise power
        to a new value corresponding to white noise RMS in uK-arcmin.
        """
        if white_noise_rms is None:
            return np.ones((2, 1))
        cnoise = np.sqrt(
            self.get_white_noise_power(tube, sky_fraction=1, units="arcmin2")
            * sky_fraction
        )
        return (white_noise_rms / cnoise)[:, None]

    def _get_requested_hitmaps(self, tube, hitmap):
        if self.homogeneous and (hitmap is None):
            ones = (
                np.ones(hp.nside2npix(self.nside))
                if self.healpix
                else pixell.enmap.ones(self.shape, self.wcs)
            )
            hitmaps = (
                np.asarray([ones, ones])
                if self.full_covariance
                else ones.reshape((1, -1))
            )
            fsky = self._sky_fraction if self._sky_fraction is not None else 1
            sky_fractions = (
                np.asarray([fsky, fsky]) if self.full_covariance else np.asarray([fsky])
            )
        else:
            hitmaps, sky_fractions = self.get_hitmaps(tube, hitmap=hitmap)

        if len(sky_fractions) == 1:
            assert hitmaps.shape[0] == 1
            fsky = np.asarray([sky_fractions[0]] * 2)
            hitmaps = np.repeat(hitmaps, 2, axis=0)
        elif len(sky_fractions) == 2:
            assert len(hitmaps) == 2
            fsky = np.asarray(sky_fractions)
        else:
            raise ValueError
        return fsky, hitmaps

    def get_noise_properties(
        self, tube, nsplits=1, hitmap=None, white_noise_rms=None, atmosphere=True
    ):
        """Get noise curves scaled with the hitmaps and the hitmaps themselves

        Parameters
        ----------
        see the docstring of simulate

        See the `band_id` attribute of the Channel class
        to identify which is the index of a Channel in the returned arrays.

        Returns
        -------
        ell : np.array
            Array of :math:`\ell`
        ps_T, ps_P : np.array
            Tube noise spectra for T and P, one row per channel, the 3rd the crosscorrelation
        fsky : np.array
            Array of sky fractions computed as <normalized N_hits>
        wnoise_power : np.array
            White noise power (high-ell limit)
        hitmaps : np.array
            Array of the hitmaps for each channel
        """
        fsky, hitmaps = self._get_requested_hitmaps(tube, hitmap)
        wnoise_scale = self._get_wscale_factor(white_noise_rms, tube, fsky)
        wnoise_power = (
            self.get_white_noise_power(tube, sky_fraction=1, units="sr")
            * nsplits
            * fsky
            * wnoise_scale.flatten()
        )
        if atmosphere:
            ell, ps_T, ps_P = self.get_fullsky_noise_spectra(
                tube, ncurve_sky_fraction=1, return_corr=True
            )
            ps_T[:2] = ps_T[:2] * fsky[:, None] * nsplits * wnoise_scale
            ps_T[2] *= np.sqrt(np.prod(ps_T[:2], axis=0))
            ps_P[:2] = ps_P[:2] * fsky[:, None] * nsplits * wnoise_scale
            ps_P[2] *= np.sqrt(np.prod(ps_P[:2], axis=0))
        else:
            ell = np.arange(self.ell_max)
            ps_T = np.zeros((3, ell.size))
            ps_T[:2] = wnoise_power[:, None] * np.ones((2, ell.size))
            ps_P = 2.0 * ps_T
        return ell, ps_T, ps_P, fsky, wnoise_power, hitmaps

    def simulate(
        self,
        tube,
        output_units="uK_CMB",
        seed=None,
        nsplits=1,
        mask_value=None,
        atmosphere=True,
        hitmap=None,
        white_noise_rms=None,
    ):
        """Create a random realization of the noise power spectrum

        Parameters
        ----------

        tube : str
            Specify a tube (for SO: ST0-ST3, LT0-LT6) see the `tubes` attribute
        output_units : str
            Output unit supported by PySM.units, e.g. uK_CMB or K_RJ
        seed : integer or tuple of integers, optional
            Specify a seed. The seed is converted to a tuple if not already
            one and appended to (0,0,6,tube_id) to avoid collisions between
            tubes, with the signal sims and with ACT noise sims, where
            tube_id is the integer ID of the tube.
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
            is generated from the white noise power in the noise model, and
            the covariance between arrays is ignored.
        hitmap : string or map, optional
            Provide the path to a hitmap to override the default used for
            the tube. You could also provide the hitmap as an array
            directly.
        white_noise_rms : float or tuple of floats, optional
            Optionally scale the simulation so that the small-scale limit white noise
            level is white_noise_rms in uK-arcmin (either a single number or
            a pair for the dichroic array).

        Returns
        -------

        output_map : ndarray or ndmap
            Numpy array with the HEALPix or CAR map realization of noise.
            The shape of the returned array is (2,3,nsplits,)+oshape, where
            oshape is (npix,) for HEALPix and (Ny,Nx) for CAR.
            The first dimension of size 2 corresponds to the two different
            bands within a dichroic tube.
            See the `band_id` attribute of the Channel class
            to identify which is the index of a Channel in the array.

            The second dimension corresponds to independent split realizations
            of the noise, e.g. it is 1 for full mission.

            The third dimension corresponds to the three polarization
            Stokes components I,Q,U

            The last dimension is the number of pixels
        """
        assert nsplits >= 1
        if mask_value is None:
            mask_value = (
                default_mask_value["healpix"]
                if self.healpix
                else default_mask_value["car"]
            )

        # This seed tuple prevents collisions with the signal sims
        # but we should eventually switch to centralized seed
        # tracking.
        if seed is not None:
            try:
                iter(seed)
            except:
                seed = (seed,)
            tube_id = self.tubes[tube][0].tube_id
            seed = (0, 0, 6, tube_id) + seed
            np.random.seed(seed)

        # In the third row we return the correlation coefficient P12/sqrt(P11*P22)
        # since that can be used straightforwardly when the auto-correlations are re-scaled.
        ell, ps_T, ps_P, fsky, wnoise_power, hitmaps = self.get_noise_properties(
            tube,
            nsplits=nsplits,
            hitmap=hitmap,
            white_noise_rms=white_noise_rms,
            atmosphere=atmosphere,
        )

        if not (atmosphere):
            if self.apply_beam_correction:
                raise NotImplementedError(
                    "Beam correction is not currently implemented for pure-white-noise sims."
                )
            # If no atmosphere is requested, we use a simpler/faster method
            # that generates white noise in real-space.
            if self.healpix:
                ashape = (hp.nside2npix(self.nside),)
                sel = np.s_[:, None, None, None]
                pmap = self.pixarea_map
            else:
                ashape = self.shape[-2:]
                sel = np.s_[:, None, None, None, None]
                pmap = pixell.enmap.enmap(self.pixarea_map, self.wcs)
            spowr = np.sqrt(wnoise_power[sel] / pmap)
            output_map = spowr * np.random.standard_normal((2, nsplits, 3) + ashape)
            output_map[:, :, 1:, :] = output_map[:, :, 1:, :] * np.sqrt(2.0)
        else:
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
                output_map = pixell.enmap.zeros((2, nsplits, 3) + self.shape, self.wcs)
                ps_T = pixell.powspec.sym_expand(np.asarray(ps_T), scheme="diag")
                ps_P = pixell.powspec.sym_expand(np.asarray(ps_P), scheme="diag")
                # TODO: These loops can probably be vectorized
                for i in range(nsplits):
                    for i_pol in range(3):
                        output_map[:, i, i_pol] = pixell.curvedsky.rand_map(
                            (2,) + self.shape,
                            self.wcs,
                            ps_T if i_pol == 0 else ps_P,
                            spin=0,
                        )

        for i in range(2):
            freq = self.tubes[tube][i].center_frequency
            if not (self.homogeneous):
                good = hitmaps[i] != 0
                # Normalize on the Effective sky fraction, see discussion in:
                # https://github.com/simonsobs/mapsims/pull/5#discussion_r244939311
                output_map[i, :, :, good] /= np.sqrt(hitmaps[i][good][..., None, None])
                output_map[i, :, :, np.logical_not(good)] = mask_value
            unit_conv = (1 * u.uK_CMB).to_value(
                u.Unit(output_units), equivalencies=u.cmb_equivalencies(freq)
            )
            output_map[i] *= unit_conv
        return output_map
