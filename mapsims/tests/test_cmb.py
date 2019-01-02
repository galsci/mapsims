import numpy as np
import healpy as hp
import os
from mapsims import cmb  # FIXME: relative path import?
from astropy.utils.data import get_pkg_data_filename


def test_load_sim():
    save_name = get_pkg_data_filename("../data/test_map.fits")
    cmb_dir = os.path.dirname(save_name)
    nside = 32
    # Make an IQU sim
    imap = cmb.get_cmb_sky(
        iteration_num=0, nside=nside, cmb_dir=cmb_dir, lensed=False, aberrated=False
    )
    imap_test = hp.read_map(save_name, field=(0, 1, 2))
    np.testing.assert_allclose(imap, imap_test)
    assert imap.shape[0] == 3
    # Make an I only sim
    imap = cmb.get_cmb_sky(
        iteration_num=0,
        nside=nside,
        has_polarization=False,
        cmb_dir=cmb_dir,
        lensed=False,
        aberrated=False,
    )
    assert imap.ndim == 2
