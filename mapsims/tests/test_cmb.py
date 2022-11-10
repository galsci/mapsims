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
    """
    mapsims/tests/data/fullskyUnlensedUnabberatedCMB_alm_set00_00000.fits
    is actually Planck_bestfit_alm_seed_583_lmax_95_K_CMB.fits from
    so_pysm_models
    """
    alm_filename = get_pkg_data_filename(
        "data/fullskyUnlensedUnabberatedCMB_alm_set00_00000.fits"
    )
    cmb_dir = os.path.dirname(alm_filename)
    nside = 32
    cmb_map = cmb.PrecomputedCMB(
        num=0,
        nside=nside,
        cmb_dir=cmb_dir,
        lensed=False,
        aberrated=False,
        input_reference_frequency=148 * u.GHz,
        input_units="uK_RJ",
    ).get_emission(148 * u.GHz)
    input_alm = hp.read_alm(alm_filename, (1, 2, 3))
    expected_cmb_map = hp.alm2map(input_alm, nside=nside) << u.uK_RJ
    assert cmb_map.shape[0] == 3
    assert_quantity_allclose(expected_cmb_map, cmb_map)


def test_standalone_cmb():
    alm_filename = get_pkg_data_filename(
        "data/fullskyUnlensedUnabberatedCMB_alm_set00_00000.fits"
    )
    cmb_dir = os.path.dirname(alm_filename)
    nside = 32
    cmb_map = cmb.StandalonePrecomputedCMB(
        num=0,
        nside=nside,
        cmb_dir=cmb_dir,
        lensed=False,
        aberrated=False,
        input_reference_frequency=148 * u.GHz,
        input_units="uK_RJ",
    ).get_emission(148 * u.GHz, fwhm=1e-5 * u.arcmin)
    input_alm = hp.read_alm(alm_filename, (1, 2, 3))
    expected_cmb_map = hp.alm2map(input_alm, nside=nside) << u.uK_RJ
    assert cmb_map.shape[0] == 3
    assert_quantity_allclose(expected_cmb_map, cmb_map)
