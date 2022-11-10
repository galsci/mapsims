import numpy as np
import healpy as hp

import pytest
from astropy.tests.helper import assert_quantity_allclose

from astropy.utils import data
import mapsims

nside = 16
res = np.deg2rad(30 / 60.0)


@pytest.mark.parametrize("tube", ["ST0", "ST3"])
def test_noise_simulator(tube):

    seed = 1234

    simulator = mapsims.SONoiseSimulator(nside=nside)
    output_map = simulator.simulate(tube, seed=seed)

    for i, ch in enumerate(simulator.tubes[tube]):
        expected_map = hp.read_map(
            data.get_pkg_data_filename(
                f"data/noise_{tube}_{ch.band}_uKCMB_classical_nside16_seed1234_healpix.fits.gz"
            ),
            (0, 1, 2),
            dtype=np.float64,
            verbose=False,
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
    assert_quantity_allclose(output_map, expected_map, rtol=1e-5)


def test_homogeneous_noise():

    seed = 1234
    tube = "ST3"

    simulator = mapsims.SONoiseSimulator(nside=nside, homogeneous=False)
    output_map = simulator.simulate(tube, seed=seed)

    assert hp.mask_bad(output_map).sum() > 100

    simulator = mapsims.SONoiseSimulator(nside=nside, homogeneous=True)
    output_map = simulator.simulate(tube, seed=seed)

    assert hp.mask_bad(output_map).sum() == 0
    assert output_map[0][0][1].std() < 4


def test_no_atmosphere():
    """Disabling atmosphere reduces power in the map"""

    seed = 1234
    tube = "ST3"

    simulator = mapsims.SONoiseSimulator(
        nside=nside, homogeneous=True, rolloff_ell=None
    )
    output_map = simulator.simulate(tube, seed=seed, atmosphere=True)

    assert output_map[0][0][0].std() > 0.5

    output_map = simulator.simulate(tube, seed=seed, atmosphere=False)

    assert output_map[0][0][0].std() < 0.5
