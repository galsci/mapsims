

import healpy, numpy as np
from pixell import enmap, curvedsky


def get_cmb_sky(iterationNum, 
                shape = None, #set these if CAR maps are desired
                wcs = None, 
                nside = None, #set this if healpix is desired
                cmbSet = 0, #We allow for more than one CMB map per lensing map
                cmbDir = '/global/cscratch1/sd/engelen/simsS1516_v0.4/data/', #FIXME change this to something from a config file or data struct!
                cmbMapType = 'LensedUnaberratedCMB' #This can also be LensedCMB, UnlensedCMB, but let's do this as the default for now.
                nFreqs = 1):
    '''
    Return a CMB map from stored alm's.  This can be in Healpix format
    (if nside is specified) or CAR format (if wcs and shape are
    specified).  The lensed alm's are pre-stored.
    '''
    nTQUs = 3
    
    filename = cmbDir + "/fullsky%s_alm_set%02d_%05d.fits" % ( cmbMaptype, cmbSet , iterationNum)

    #The hdu = (1, 2,3) means get all of T, E, B
    #Note the alm's are stored as complex32, so upgrade this for processing
    almTeb = np.complex128(healpy.fitsfunc.read_alm(filename, hdu = (1,2,3)))

    #Here we can multiply the alms by the appropriate beam;
    #healpy.sphtfunc.almxfl can be used.  Not included yet.  Once we
    #do, we will have to call the inverse SHT nFreqs times, as in actsims.
    
    if nside is not None:

        #Then we are outputting a healpix map
        output = healpy.alm2map(almTeb, nside)

        #Here we want to multiply the map by the modulation factor.  FIXME: not implemented yet
        

        return np.tile(output, (nFreqs, 1))
    
    elif (wcs is not None and shape is not None):

        mapTqu = enmap.empty( (nFreqs,nTQUs,)+shape[-2:], wcs)

        curvedsky.alm2map(almTeb, mapTqu, spin = [0, 2], verbose = True)

        #Tile this to return something of shape (nFreqs, 3, Ny, Nx)
        #Why do we need to return this nFreqs times? Because in the future we will multiply by a frequency-dependent modulation factor
        output = enmap.ndmap(np.tile(mapTqu, (nFreqs, 1, 1, 1)), wcs)

        

        #Here we want to multiply the map by the modulation factor.  FIXME: not implemented yet
        return output

    else:
        raise ValueError("You must specify either nside or both of shape and wcs")
    
