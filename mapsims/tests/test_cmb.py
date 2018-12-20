import numpy as np
import os,sys
import healpy as hp
from mapsims import cmb # FIXME: relative path import?
from astropy.utils.data import get_pkg_data_filename

def test_load_sim():
    path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    cmb_dir = path+'/data' # FIXME: better way to get path to data?

    nside = 32
    # Make an IQU sim
    imap = cmb.get_cmb_sky(iteration_num=0,nside = nside,cmb_dir=cmb_dir,lensed=False,aberrated=False)
    assert imap.shape[0]==3
    # Make an I only sim
    imap = cmb.get_cmb_sky(iteration_num=0,nside = nside, has_polarization=False,cmb_dir=cmb_dir,aberrated=False)
    assert imap.ndim==2

