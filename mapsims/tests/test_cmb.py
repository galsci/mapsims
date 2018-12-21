import numpy as np
import os,sys
import healpy as hp
from mapsims import cmb # FIXME: relative path import?
from astropy.utils.data import get_pkg_data_filename

def test_load_sim():
    cmb_dir = 'mapsims/data' # FIXME: better way to get path to data?
    save_name = 'mapsims/data/test_map.fits'
    nside = 32
    # Make an IQU sim
    imap = cmb.get_cmb_sky(iteration_num=0,nside = nside,cmb_dir=cmb_dir,lensed=False,aberrated=False)
    imap_test = hp.read_map(save_name,field=(0,1,2))
    np.testing.assert_allclose(imap, imap_test)
    assert imap.shape[0]==3
    # Make an I only sim
    imap = cmb.get_cmb_sky(iteration_num=0,nside = nside, has_polarization=False,cmb_dir=cmb_dir,lensed=False,aberrated=False)
    assert imap.ndim==2

