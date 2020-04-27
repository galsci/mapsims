import os.path
import numpy as np
import healpy as hp
from astropy.utils import data
import warnings

import pysm.units as u

from so_models_v3 import SO_Noise_Calculator_Public_v3_1_1 as so_models
from . import so_utils
from . import utils as mutils

# pixell is optional and needed when CAR simulations are requested
try:
    from pixell import enmap, wcsutils, curvedsky, powspec
except: pass

sensitivity_modes = {"baseline": 1, "goal": 2}
one_over_f_modes = {"pessimistic": 0, "optimistic": 1}
default_mask_value = {"healpix": hp.UNSEEN, "car": np.nan}
_hitmap_version = "v0.2"

def _band_ids_from_tube(tube):
    """Internal function to convert a tube name
    to a pair of indices, where the indices
    correspond to the positions in a list of
    frequencies that the two bands of the
    tube correspond to. The list has the
    same order that the noise curve code
    uses and is identical to the unique
    set of frequencies in so_utils.frequencies.

    e.g.
    >>> _band_ids_from_tube('LT0')
    [6, 7]
    >>> _band_ids_from_tube('ST1')
    [2, 3]
    >>> _band_ids_from_tube('ST2')
    [4, 5]
    
    """
    tubes = so_utils.tubes
    bands = tubes[tube]
    freqs = [so_utils.band_freqs[x] for x in bands]
    available_frequencies = np.unique(so_utils.frequencies)
    band_ids = [available_frequencies.searchsorted(f) for f in freqs]
    return band_ids

def _band_index(tube,band):
    """
    Internal function to get the index position
    of a band in a tube.

    e.g.
    >>> _band_index('LT0','UHF1')
    0
    >>> _band_index('LT0','UHF2')
    1
    >>> _band_index('ST2','MFS1')
    0
    >>> _band_index('ST2','MFS2')
    1
    """
    return so_utils.tubes[tube].index(band)


class SONoiseSimulator:
    def __init__(
        self,
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
        rolloff_ell=50,
        survey_efficiency=0.2,
        full_covariance=True,
        LA_years=5,
        LA_noise_model="SOLatV3point1",
        elevation=50,
        SA_years=5,
        SA_one_over_f_mode="pessimistic",
        sky_fraction = None,
        cache_hitmaps = True,
        boolean_sky_fraction = True,
        rescale_white_noise = None,
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
            shape of ndmap array (see pixell.enmap). Must also specify wcs.
        wcs : astropy.wcs.wcs.WCS instance
            World Coordinate System for geometry of map (see pixell.enmap). Must
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
            If homogenous is True, this sky_fraction is used for the noise curves.
        cache_hitmaps : bool
            If True, caches hitmaps.
        boolean_sky_fraction: bool
            If True, determines fsky based on fraction of hitmap that is zero. If False,
            determines sky_fraction from <Nhits>.
        rescale_white_noise : float or tuple of floats, optional
        
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
            self.pmap = enmap.pixsizemap(self.shape, self.wcs)
        else:
            assert shape is None
            assert wcs is None
            self.healpix = True
            self.nside = nside
            self.ell_max = ell_max if ell_max is not None else 3 * nside

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
        self.homogenous = homogenous

        self.hitmap_version = _hitmap_version
        self._cache = cache_hitmaps
        self._hmap_cache = {}
        self.remote_data = mutils.RemoteData(
            healpix=self.healpix, version=self.hitmap_version
        )


    def get_beam_fwhm(self, tube, band=None):
        """Get beam FWHMs in arcminutes corresponding to the tueb.
        This is useful if non-beam-deconvolved sims are requested and you want to 
        know what beam to apply to your signal simulation.

        Parameters
        ----------

        tube : str
            Specify a specific tube. For available
            tubes and their channels, see so_utils.tubes.

        band : str,optional
            Optionally specify the band name within the tube to get just its
            white noise.


        Returns
        -------

        beam : tuple of floats
            The beam FWHM in arcminutes either as
            a tuple for the pair of bands in the tube, or just for the specific
            band requested.
        """

        survey = self._get_survey(tube)
        bands = _band_ids_from_tube(tube)
        ret = survey.get_beams()[bands]
        if band is not None: ret = ret[_band_index(tube,band)]
        return ret

    def _get_survey(self,tube):
        """Internal function to get the survey object
        from the SO noise model code.
        """
        telescope = f'{tube[0]}A' # get LA or SA from tube name
        if telescope=='SA':
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
        elif telescope=='LA':
            survey = getattr(so_models, self.LA_noise_model)(
                sensitivity_mode=self.sensitivity_mode,
                survey_efficiency=self.survey_efficiency,
                survey_years=self.LA_years,
                N_tubes=[1,1,1],
                el=self.elevation,
            )
        return survey


    def get_noise_spectra(self, tube, ncurve_fsky=1):
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
        survey = self._get_survey(tube)
        if telescope == "SA":
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
            ell, noise_ell_T, noise_ell_P = survey.get_noise_curves(
                ncurve_fsky,  # We load hitmaps later, so we compute and apply sky fraction later
                self.ell_max,
                delta_ell=1,
                full_covar=True,
                deconv_beam=self.apply_beam_correction,
                rolloff_ell=self.rolloff_ell,
            )

        assert ell[0] == 2 # make sure the noise code is still returning something 
        # that starts at ell=2
        ls = np.arange(ell.size + 2)
        nells_T = np.zeros((3,ell.size + 2))
        nells_P = np.zeros((3,ell.size + 2))
        b1,b2 = _band_ids_from_tube(tube)
        for n_out,n_in in zip([nells_T,nells_P],[noise_ell_T,noise_ell_P]):
            n_out[0,2:] = n_in[b1][b1]
            n_out[1,2:] = n_in[b2][b2]
            n_out[2,2:] = n_in[b1][b2] if not(self.full_covariance) else 0
            if self.no_power_below_ell is not None:
                n_out[:,ls < self.no_power_below_ell] = 0

        return ls,nells_T,nells_P

    def _validate_map(self,fmap):
        """Internal function to validate an externally provided map.
        It checks the healpix or CAR attributes against what the
        class was initialized with. It adds a leading dimension if 
        necessary.
        """
        shape = fmap.shape
        if self.healpix:
            if len(shape)==1: 
                npix = shape[0]
            elif len(shape)==2:
                assert shape[0]==1 or shape[0]==2
                npix = shape[1]
            else:
                raise ValueError
            assert npix == hp.nside2npix(self.nside)
            if len(shape)==1: 
                return fmap[None,:]
            else: 
                return fmap
        else:
            assert wcsutils.is_compatible(fmap.wcs, self.wcs)            
            if len(shape)==2:
                ashape = shape
            elif len(shape)==3:
                assert shape[0]==1 or shape[0]==2
                ashape = shape[-2:]
            else:
                raise ValueError
            assert [x==y for x,y in zip(ashape,self.shape)]
            if len(shape)==2:
                return fmap[None,...]
            else:
                return fmap
                

    def _load_map(self,fname,**kwargs):
        """Internal function to load a healpix or CAR map
        from disk and reproject it if necessary.
        """
        # If not a string try as a healpix or CAR map
        if not(isinstance(fname,str)): return self._validate_map(fname)
        
        # Check if in cache
        if self._cache: 
            try: return self._hmap_cache[fname]
            except: pass

        # else load
        if self.healpix:
            hitmap = hp.ud_grade(
                    hp.read_map(fname, verbose=False), nside_out=self.nside
                )
        else:

            hitmap = enmap.read_map(fname)
            if wcsutils.is_compatible(hitmap.wcs, self.wcs):
                hitmap = enmap.extract(hitmap, self.shape, self.wcs)
            else:
                warnings.warn(
                    "WCS of hitmap with nearest pixel-size is not compatible, so interpolating hitmap"
                )
                hitmap = enmap.project(hitmap, self.shape, self.wcs,order=0) 

        # and then cache and return
        if self._cache: self._hmap_cache[fname] = hitmap
        return hitmap


    def _process_hitmaps(self,hitmaps):
        """Internal function to process hitmaps and based on the
        desired scheme, obtain sky fractions from them.
        """
        nhitmaps = hitmaps.shape[0]
        assert nhitmaps==1 or nhitmaps==2
        if self.boolean_sky_fraction:
            for i in range(nhitmaps):
                hitmaps[i] /= hitmaps[i].max()

            if self.healpix:
                sky_fractions = [(hitmaps[i] != 0).sum() / hitmaps[i].size for i in range(nhitmaps)]
            else:
                pmap = self.pmap
                sky_fractions = [(
                    pmap[hitmaps[i] != 0].sum()
                    / 4.0
                    / np.pi
                ) for i in range(nhitmaps)]
        else:
            raise NotImplementedError
        return hitmaps, sky_fractions
            
            

    def get_hitmaps(self, tube=None, hitmap=None):
        """Get and process hitmaps and sky fractions for the provided tube or provided
        an external one.

        Parameters
        ----------

        tube : str
            Specify a specific tube. For available
            tubes and their channels, see so_utils.tubes.

        hitmap : string or map, optional
            Provide the path to a hitmap to override the default used for 
            the tube. You could also provide the hitmap as an array
            directly.

        Returns
        -------

        hitmaps : ndarray or ndmap
            Processed hitmaps.

        sky_fractions : float
            The sky fraction covered by the survey determined from the hitmaps.

        """

        if hitmap is not None: return self._process_hitmaps(self._load_map(hitmap))

        telescope = f'{tube[0]}A' # get LA or SA from tube name

        if not (self.healpix):
            npixheight = min(
                {"LA": [0.5, 2.0], "SA": [4.0, 12.0]}[telescope],
                key=lambda x: abs(x - self._pixheight),
            )
            car_suffix = f"_CAR_{npixheight:.2f}_arcmin"
        else:
            car_suffix = ""

        bands = so_utils.tubes[tube]

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

    def get_white_noise_power(self, tube, sky_fraction,band=None,units='sr'):
        """Get white noise power in uK^2-sr (units='sr') or
        uK^2-arcmin^2 (units='arcmin2') corresponding to the tube name tube.
        This is useful if you want to generate your own simulations that do not
        have the atmospheric component.

        Parameters
        ----------

        tube : str
            Specify a specific tube. For available
            tubes and their channels, see so_utils.tubes.

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
            

        """
        survey = self._get_survey(tube)
        bands = _band_ids_from_tube(tube)
        ret = survey.get_white_noise(sky_fraction, units=units)[bands]
        if band is not None: ret = ret[_band_index(tube,band)]
        return ret

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

        tube : str
            Specify a specific tube. For available
            tubes and their channels, see so_utils.tubes.
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

        Returns
        -------

        output_map : ndarray or ndmap
            Numpy array with the HEALPix or CAR map realization of noise.
            The shape of the returned array is (2,3,nsplits,)+oshape, where 
            oshape is (npix,) for HEALPix and (Ny,Nx) for CAR.
            The first dimension of size 2 corresponds to the two different
            bands within a dichroic tube. The second dimension corresponds
            the three polarization Stokes components I,Q,U and the third
            dimension corresponds to independent split realizations of the
            noise.
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
            try: iter(seed)
            except: seed = (seed,)
            tube_id = so_utils.tube_names.index(tube)
            seed = (0,0,6,tube_id) + seed
            np.random.seed(seed)

        if self.homogenous and (hitmap is None):
            ones = np.ones(hp.nside2npix(self.nside)) if self.healpix else enmap.ones(self.shape,self.wcs)
            hitmaps = [ones, ones] if self.full_covariance else ones
            fsky = self._sky_fraction if self._sky_fraction is not None else 1
            sky_fractions = [fsky, fsky] if self.full_covariance else fsky
        else:
            hitmaps, sky_fractions = self.get_hitmaps(tube,hitmap=hitmap)

        if len(sky_fractions)==1: fsky = np.asarray([sky_fractions[0]]*3)
        else: fsky = np.append(sky_fractions,[np.mean(sky_fractions)])

        if not(atmosphere):
            # If no atmosphere is requested, we use a simpler/faster method
            # that generates white noise in real-space.
            npower = self.get_white_noise_power(tube,sky_fraction=1,units='arcmin2') * nsplits * fsky[:2]
            if self.healpix:
                ashape = (hp.nside2npix(self.nside),)
                sel = np.s_[:,None,None,None]
                pmap = (4.*np.pi / hp.nside2npix(self.nside))*((180.*60./np.pi)**2.)
            else:
                ashape = self.shape[-2:]
                sel = np.s_[:,None,None,None,None]
                pmap = self.pmap*((180.*60./np.pi)**2.)
            spowr = (np.sqrt(npower[sel]/pmap))
            output_map = spowr*np.random.standard_normal((2,nsplits,3,)+ashape)
            output_map[:,:,1:,:] = output_map[:,:,1:,:] * np.sqrt(2.)
        else:
            ls,ps_T,ps_P = self.get_noise_spectra(tube, ncurve_fsky=1)
            ps_T = ps_T * fsky[:,None] * nsplits
            ps_P = ps_P * fsky[:,None] * nsplits
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
                output_map = np.zeros((2, nsplits, 3, ) + self.shape)
                ps_T = powspec.sym_expand(np.asarray(ps_T), scheme="diag")
                ps_P = powspec.sym_expand(np.asarray(ps_P), scheme="diag")
                # TODO: These loops can probably be vectorized
                for i in range(nsplits):
                    for i_pol in range(3):
                        output_map[:,i,i_pol] = curvedsky.rand_map((2,) + self.shape, self.wcs, ps_T if i_pol==0 else ps_P,spin=0)

        tubes = so_utils.tubes
        bands = tubes[tube]
        telescope = f'{tube[0]}A' # get LA or SA from tube name

        for out_map, hitmap, sky_fraction, band in zip(
            output_map, hitmaps, sky_fractions, bands
        ):
            freq = so_utils.SOChannel(telescope, band, tube=tube).center_frequency
            good = hitmap != 0
            # Normalize on the Effective sky fraction, see discussion in:
            # https://github.com/simonsobs/mapsims/pull/5#discussion_r244939311
            out_map[:, :, good] /= np.sqrt(hitmap[good] / hitmap.mean() * sky_fraction)
            out_map[:, :, np.logical_not(good)] = mask_value
            unit_conv = (1 * u.uK_CMB).to_value(
                u.Unit(output_units),
                equivalencies=u.cmb_equivalencies(freq),
            )
            out_map *= unit_conv

        return output_map
