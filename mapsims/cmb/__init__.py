import healpy as hp, numpy as np
import so_pysm_models as spm


def get_cmb_sky(
    iteration_num,
    nside=None,  # set this if healpix is desired
    shape=None,  # set shape and wcs if CAR maps are desired
    wcs=None,
    lensed=True,
    aberrated=False,
    modulation=False,
    has_polarization=True,
    cmb_set=0,  # We allow for more than one CMB map per lensing map
    cmb_dir=None,
    nu=None,
    input_units="uK_RJ",
    pixel_indices=None,
):
    """
    Return a CMB map from stored alm's.  This can be in Healpix format
    (if nside is specified) or rectangular pixel format (if wcs and shape are
    specified).  The lensed alm's are pre-stored.
    If rectangular, it returns a stack of enmaps of shape (nfreqs, ncomp, ny, nx). 
    If Healpix, it will return a numpy array of shape (nfreqs, ncomp, npix) 

    Args:
        iteration_num: integer specifying which sim iteration to use
        nside: nside of healpix map to project alms to. If None, uses 
        rectangular pixel geometry specified through shape and wcs.
        shape: shape of ndmap array (see pixell.enmap)
        wcs: World Coordinate System for geometry of map (see pixell.enmap)
        lensed: whether to load lensed or unlensed sims
        aberrated: whether to load aberrated or unaberrated sims
        pol: if True, return ncomp=3 I,Q,U components, else just ncomp=1 I
        cmb_set: integer specifying which set of sims to use
        cmb_dir: override the default lensed alm directory path
        nfreqs: number of copies of the CMB sky to provide. When modulation
        is implemented, this argument will be changed to a frequency bandpass
        specification that applies frequency-dependent modulation to the sims.

    Returns:
        output: (nfreqs,ncomp,npix) healpix array if nside is specified, else
        returns (nfreqs,ncomp,Ny,Nx) rectangular pixel ndmap.
        
    """
    ncomp = 3 if has_polarization else 1
    filename = _get_cmb_map_string(cmb_dir, iteration_num, cmb_set, lensed, aberrated)
    # The hdu = (1, 2,3) means get all of T, E, B
    # Note the alm's are stored as complex32, so upgrade this for processing

    sgen = spm.PrecomputedAlms(
        filename,
        target_nside=nside,
        target_shape=shape,
        target_wcs=wcs,
        input_units=input_units,
        has_polarization=has_polarization,
        pixel_indices=pixel_indices,
    )

    return sgen.signal(nu=nu, modulation=modulation)


def _get_default_cmb_directory():
    # FIXME: remove hard-coding to use preferred directory path system
    return "/global/project/projectdirs/sobs/v4_sims/mbs/cmb"


def _get_cmb_map_string(cmb_dir, iteration_num, cmb_set, lensed, aberrated):
    # Implements the CMB lensed alms file naming convention
    # Ideally the same function should be used when saving sims
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
