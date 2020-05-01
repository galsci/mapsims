import numpy as np
import healpy as hp

import pytest
from astropy.tests.helper import assert_quantity_allclose

from astropy.utils import data
import mapsims
import pysm.units as u
from mapsims import so_utils

nside = 16
res = np.deg2rad(30 / 60.0)


def test_freq_order():
    freqs = np.unique(so_utils.frequencies)
    ifreqs = (27, 39, 93, 145, 225, 280)
    for f, f2 in zip(freqs, ifreqs):
        assert f == f2


@pytest.mark.parametrize("tube", ["ST0", "ST3"])
def test_noise_simulator(tube):

    seed = 1234

    simulator = mapsims.SONoiseSimulator(nside=nside)
    output_map = simulator.simulate(tube, seed=seed)

    for i, band in enumerate(so_utils.tubes[tube]):
        expected_map = hp.read_map(
            data.get_pkg_data_filename(
                f"data/noise_{tube}_{band}_uKCMB_classical_nside16_seed1234_healpix.fits.gz"
            ),
            (0, 1, 2),
        )
        assert_quantity_allclose(output_map[i, 0], expected_map)


@pytest.mark.parametrize("tube", ["LT0", "ST3"])
def test_noise_simulator_car(tube):

    from pixell import enmap

    seed = 1234
    shape, wcs = enmap.fullsky_geometry(res=res)
    simulator = mapsims.SONoiseSimulator(shape=shape, wcs=wcs)

    output_map = simulator.simulate(tube, seed=seed)
    expected_map = enmap.read_map(
        data.get_pkg_data_filename(
            f"data/noise_{tube}_uKCMB_classical_res30_seed1234_car.fits.gz"
        )
    )
    assert_quantity_allclose(output_map, expected_map)
