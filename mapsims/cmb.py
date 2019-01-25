import so_pysm_models


class SOPrecomputedCMB(so_pysm_models.PrecomputedAlms):

    def __init__(
        self,
        iteration_num,
        nside=None,  # set this if healpix is desired
        shape=None,  # set shape and wcs if CAR maps are desired
        wcs=None,
        lensed=True,
        aberrated=False,
        has_polarization=True,
        cmb_set=0,  # We allow for more than one CMB map per lensing map
        cmb_dir=None,
        input_units="uK_RJ",
        pixel_indices=None,
    ):
        """
        Return a CMB map from stored alm's.  This can be in HEALPix format
        (if nside is specified) or rectangular pixel format (if wcs and shape are
        specified).  The lensed alm's are pre-stored.
        If rectangular, it returns a stack of enmaps of shape (nfreqs, ncomp, ny, nx). 
        If HEALPix, it will return a numpy array of shape (nfreqs, ncomp, npix) 

        Parameters
        ----------
        iteration_num : int
            integer specifying which sim iteration to use
        nside : int
            nside of healpix map to project alms to. If None, uses 
            rectangular pixel geometry specified through shape and wcs.
        shape : tuple of ints
            shape of ndmap array (see pixell.enmap)
        wcs : str
            World Coordinate System for geometry of map (see pixell.enmap)
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

        filename = _get_cmb_map_string(
            cmb_dir, iteration_num, cmb_set, lensed, aberrated
        )

        super().__init__(
            filename,
            target_nside=nside,
            target_shape=shape,
            target_wcs=wcs,
            input_units=input_units,
            has_polarization=has_polarization,
            pixel_indices=pixel_indices,
        )


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
