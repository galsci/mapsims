from pixell import enmap
import numpy as np
import os,sys
import healpy as hp
from mapsims import cmb # FIXME: relative path import?

def test_load_sim():
    nside = 32
    # Make an IQU sim
    imap = cmb.get_cmb_sky(iteration_num=0,nside = nside)
    assert imap.shape[0]==3
    # Make an I only sim
    imap = cmb.get_cmb_sky(iteration_num=0,nside = nside, pol=False)
    assert imap.ndim==1

