import numpy as np
from astropy.tests.helper import assert_quantity_allclose
import healpy as hp

import pysm.units as u
from astropy.utils import data

import mapsims
import so_pysm_models

NSIDE = 16


def test_from_config_v02():

    simulator = mapsims.from_config(
        data.get_pkg_data_filename("data/example_config_v0.2.toml", package="mapsims")
    )
    output_map = simulator.execute(write_outputs=False)[simulator.channels[0][0].tag]

    expected_map = hp.read_map(
        data.get_pkg_data_filename("data/simonsobs_ST0_UHF1_nside16.fits.gz"), (0, 1, 2)
    )
    assert_quantity_allclose(output_map, expected_map, rtol=1e-2)
