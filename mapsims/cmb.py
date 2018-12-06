import healpy, numpy as np
from pixell import enmap, curvedsky

def get_cmb_sky(iteration_num, 
                nside = None, #set this if healpix is desired
                shape = None, #set shape and wcs if CAR maps are desired
                wcs = None,
                lensed = True,
                aberrated = False,
                pol = True,
                cmb_set = 0, #We allow for more than one CMB map per lensing map
                cmb_dir = None, 
                nfreqs = 1):
    '''
    Return a CMB map from stored alm's.  This can be in Healpix format
    (if nside is specified) or CAR format (if wcs and shape are
    specified).  The lensed alm's are pre-stored.
    If CAR, it returns a stack of enmaps of shape (nfreqs, ncomp, ny, nx). 
    If Healpix, it will return a numpy array of shape (nfreqs, ncomp, npix) 
    '''
    ncomp = 3 if pol else 1
    filename = _get_cmb_map_string(cmb_dir,iteration_num,cmb_set,lensed,aberrated)
    #The hdu = (1, 2,3) means get all of T, E, B
    #Note the alm's are stored as complex32, so upgrade this for processing
    alm_teb = np.complex128(healpy.fitsfunc.read_alm(filename, hdu = (1,2,3) if pol else 1))
    #Here we can multiply the alms by the appropriate beam;
    #healpy.sphtfunc.almxfl can be used.  Not included yet.  Once we
    #do, we will have to call the inverse SHT nfreqs times, as in actsims.
    if nside is not None:
        #Then we are outputting a healpix map
        map_tqu = healpy.alm2map(alm_teb, nside)
        output = np.tile(map_tqu, (nfreqs, 1, 1)) if nfreqs>1 else map_tqu
        #Here we want to multiply the map by the modulation factor.  FIXME: not implemented yet
    elif (wcs is not None and shape is not None):
        map_tqu = enmap.empty( (ncomp,)+shape[-2:], wcs)
        curvedsky.alm2map(alm_teb, map_tqu, spin = [0, 2], verbose = True)
        #Tile this to return something of shape (nfreqs, 3, Ny, Nx)
        #Why do we need to return this nfreqs times? Because in the future we will multiply by a frequency-dependent modulation factor
        output = enmap.ndmap(np.tile(mapTqu, (nfreqs, 1, 1, 1)), wcs) if nfreqs>1 else map_tqu
        #Here we want to multiply the map by the modulation factor.  FIXME: not implemented yet
    else:
        raise ValueError("You must specify either nside or both of shape and wcs")
    return output

def  _get_default_cmb_directory():
    #FIXME: remove hard-coding to use preferred directory path system
    return "/global/project/projectdirs/sobs/v4_sims/mbs/cmb/"

def _get_cmb_map_string(cmb_dir,iteration_num,cmb_set,lensed,aberrated):
    # Implements the CMB lensed alms file naming convention
    # Ideally the same function should be used when saving sims
    if cmb_dir is None: cmb_dir = _get_default_cmb_directory()
    lstring = "Lensed" if lensed else "Unlensed"
    abstring = "Aberrated" if aberrated else "Unaberrated"
    cmb_map_type = "%s%sCMB" % (lstring,abstring)
    filename = cmb_dir + "/fullsky%s_alm_set%02d_%05d.fits" % ( cmb_map_type, cmb_set , iteration_num)
    return 

    
