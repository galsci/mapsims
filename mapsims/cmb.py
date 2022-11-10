import healpy as hp
import numpy as np
from .alms import PrecomputedAlms


class PrecomputedCMB(PrecomputedAlms):
    def __init__(
        self,
        num,
        nside=None,  # set this if healpix is desired
        shape=None,  # set shape and wcs if CAR maps are desired
        wcs=None,
        lensed=True,
        aberrated=False,
        has_polarization=True,
        cmb_set=0,  # We allow for more than one CMB map per lensing map
        cmb_dir=None,
        input_units="uK_CMB",
        input_reference_frequency=None,
        precompute_output_map=False,
        map_dist=None,
    ):
        """
        Return a CMB map from stored alm's.  This can be in HEALPix format
        (if nside is specified) or rectangular pixel format (if wcs and shape are
        specified).  The lensed alm's are pre-stored.
        If rectangular, it returns a stack of enmaps of shape (nfreqs, ncomp, ny, nx).
        If HEALPix, it will return a numpy array of shape (nfreqs, ncomp, npix)

        Parameters
        ----------
        num : int
            integer specifying which sim iteration to use
        input_units : string
            Input unit strings as defined by pysm.convert_units, e.g. K_CMB, uK_RJ, MJysr
        input_reference_frequency_GHz : float
            If input units are K_RJ or Jysr, the reference frequency
        nside : int
            nside of healpix map to project alms to. If None, uses
            rectangular pixel geometry specified through shape and wcs.
        shape : tuple of ints
            shape of ndmap array (see pixell.enmap). Must also specify wcs.
        wcs : astropy.wcs.wcs.WCS insance
            World Coordinate System for geometry of map (see pixell.enmap). Must
            also specify shape.
        lensed : bool
            Whether to load lensed or unlensed sims
        aberrated : bool
            Whether to load aberrated or unaberrated sims
        pol : bool
            If True, return ncomp=3 I,Q,U components, else just ncomp=1 I
        cmb_set : int
            Integer specifying which set of sims to use
        cmb_dir : path
            Override the default lensed alm directory path
        nfreqs : int
            Number of copies of the CMB sky to provide. When modulation
            is implemented, this argument will be changed to a frequency bandpass
            specification that applies frequency-dependent modulation to the sims.

        Returns
        -------
        output
            (nfreqs,ncomp,npix) HEALPix numpy array if nside is specified, else
            returns (nfreqs,ncomp,Ny,Nx) rectangular pixel ndmap.
        """

        filename = _get_cmb_map_string(cmb_dir, num, cmb_set, lensed, aberrated)

        super().__init__(
            filename,
            nside=nside,
            target_shape=shape,
            target_wcs=wcs,
            input_units=input_units,
            input_reference_frequency=input_reference_frequency,
            has_polarization=has_polarization,
            map_dist=map_dist,
            precompute_output_map=precompute_output_map,
        )


class StandalonePrecomputedCMB(PrecomputedAlms):
    def __init__(
        self,
        num,
        nside=None,  # set this if healpix is desired
        shape=None,  # set shape and wcs if CAR maps are desired
        wcs=None,
        lensed=True,
        aberrated=False,
        has_polarization=True,
        cmb_set=0,  # We allow for more than one CMB map per lensing map
        cmb_dir=None,
        input_units="uK_CMB",
        input_reference_frequency=None,
        map_dist=None,
    ):
        """
        Equivalent of PrecomputedCMB to be executed outside of PySM.
        This is useful if you are not simulating any other component with PySM.
        It loads the Alms in the constructor. When `simulate(ch)` is called,
        it convolves the Alms with the beam, generates a map and applies unit
        conversion. The lensing potential map corresponding to the simulation
        can be obtained by calling `get_phi_alm()`.
        """

        filename = _get_cmb_map_string(cmb_dir, num, cmb_set, lensed, aberrated)

        self.iteration_num = num
        self.cmb_dir = cmb_dir

        super().__init__(
            filename,
            nside=nside,
            target_shape=shape,
            target_wcs=wcs,
            input_units=input_units,
            input_reference_frequency=input_reference_frequency,
            has_polarization=has_polarization,
            map_dist=map_dist,
            precompute_output_map=False,
        )

    def get_phi_alm(self):
        """Return the lensing potential (phi) alms corresponding to this sim"""
        return hp.read_alm(_get_phi_map_string(self.cmb_dir, self.iteration_num))

    def simulate(self, ch, output_units="uK_CMB"):
        """Return a simulated noise map for a specific Simons Observatory channel

        Parameters
        ----------
        ch : mapsims.Channel
            Simons Observatory Channel object
        output_units : str
            Units as defined by `pysm.convert_units`, e.g. uK_CMB or K_RJ
        """

        def _wrap_wcs(x):
            if self.wcs is not None:
                from pixell import enmap

                return enmap.enmap(x, self.wcs)
            else:
                return x

        if isinstance(ch, tuple):
            if self.nside is None:
                raise NotImplementedError("Tube simulations for CAR not supported yet")
            output_map = np.zeros(
                (len(ch), 3, hp.nside2npix(self.nside)), dtype=np.float64
            )
            for i, each in enumerate(ch):
                output_map[i] = _wrap_wcs(
                    self.get_emission(
                        freqs=each.center_frequency,
                        fwhm=each.beam,
                        output_units=output_units,
                    )
                )
            return output_map
        else:
            return _wrap_wcs(
                self.get_emission(
                    freqs=ch.center_frequency, fwhm=ch.beam, output_units=output_units
                )
            )


def _get_default_cmb_directory():
    # FIXME: remove hard-coding to use preferred directory path system
    return "/global/project/projectdirs/sobs/v4_sims/mbs/cmb"


def _get_cmb_map_string(cmb_dir, iteration_num, cmb_set, lensed, aberrated):
    """Implements the CMB lensed alms file naming convention
    Ideally the same function should be used when saving sims
    """
    if cmb_dir is None:
        cmb_dir = _get_default_cmb_directory()
    lstring = "Lensed" if lensed else "Unlensed"
    abstring = "Abberated" if aberrated else "Unabberated"
    cmb_map_type = "%s%sCMB" % (lstring, abstring)
    filename = cmb_dir + "/fullsky%s_alm_set%02d_%05d.fits" % (
        cmb_map_type,
        cmb_set,
        iteration_num,
    )
    return filename


def _get_phi_map_string(cmb_dir, iteration_num):
    """Implements the lensing potential alms file naming convention
    Ideally the same function should be used when saving sims
    """
    if cmb_dir is None:
        cmb_dir = _get_default_cmb_directory()
    filename = cmb_dir + "/input_phi/fullskyPhi_alm_%05d.fits" % (iteration_num,)
    return filename
