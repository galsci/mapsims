import numpy as np
import healpy as hp

import pytest

from astropy.utils import data
from mapsims import noise

NSIDE = 16


@pytest.mark.parametrize("telescope", ["SA", "LA"])
def test_noise_simulator(telescope):

    simulator = noise.SONoiseSimulator(
        telescope=telescope,
        band=27,
        nside=NSIDE,
        return_uK_CMB=True,
        sensitivity_mode="baseline",
        apply_beam_correction=True,
        apply_kludge_correction=True,
        scanning_strategy="classical",
        LA_number_LF=1,
        LA_number_MF=4,
        LA_number_UHF=2,
        SA_years_LF=1,
        SA_one_over_f_mode="pessimistic",
    )

    output_map = simulator.simulate(seed=100)
    expected_map = hp.read_map(
        data.get_pkg_data_filename(
            "data/{}_noise_classical_seed100.fits.gz".format(telescope)
        ),
        (0, 1, 2),
    )
    np.testing.assert_allclose(output_map, expected_map)
