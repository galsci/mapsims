import os

import healpy as hp

from astropy.utils.data import get_pkg_data_filename

try:  # PySM >= 3.2.1
    import pysm3.units as u
except ImportError:
    import pysm.units as u

from .. import cmb

from astropy.tests.helper import assert_quantity_allclose


def test_load_sim():
    save_name = get_pkg_data_filename("data/test_map.fits")
    cmb_dir = os.path.dirname(save_name)
    nside = 32
    # Make an IQU sim
    imap = cmb.SOPrecomputedCMB(
        num=0,
        nside=nside,
        cmb_dir=cmb_dir,
        lensed=False,
        aberrated=False,
        input_reference_frequency=148 * u.GHz,
        input_units="uK_RJ",
    ).get_emission(148 * u.GHz)
    imap_test = hp.read_map(save_name, field=(0, 1, 2)) << u.uK_RJ
    assert_quantity_allclose(imap, imap_test)
    assert imap.shape[0] == 3
    # Make an I only sim
    imap = cmb.SOPrecomputedCMB(
        num=0,
        nside=nside,
        has_polarization=False,
        cmb_dir=cmb_dir,
        lensed=False,
        aberrated=False,
        input_units="uK_RJ",
        input_reference_frequency=148 * u.GHz,
    ).get_emission(148 * u.GHz)


def test_standalone_cmb():

    save_name = get_pkg_data_filename("data/test_map.fits")
    cmb_dir = os.path.dirname(save_name)
    nside = 32
    freq = 145 * u.GHz
    # Make an IQU sim
    imap = cmb.SOStandalonePrecomputedCMB(
        num=0,
        nside=nside,
        cmb_dir=cmb_dir,
        lensed=False,
        aberrated=False,
        input_units="uK_RJ",
        input_reference_frequency=freq,
    ).get_emission(freq, fwhm=1e-5 * u.arcmin)
    imap_test = hp.read_map(save_name, field=(0, 1, 2)) << u.uK_RJ
    assert_quantity_allclose(imap, imap_test)
