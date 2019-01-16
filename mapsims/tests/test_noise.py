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
        band=93,
        nside=NSIDE,
        ell_max=500,
        return_uK_CMB=True,
        sensitivity_mode="baseline",
        apply_beam_correction=True,
        apply_kludge_correction=True,
        scanning_strategy="classical",
        LA_number_LF=1,
        LA_number_MF=4,
        LA_number_UHF=2,
        SA_years_LF=1,
        SA_one_over_f_mode="optimistic",
        seed=1234,
    )

    output_map = simulator.simulate()
    expected_map = hp.read_map(
        data.get_pkg_data_filename(
            "data/noise_{}_uKCMB_classical_nside16_channel2_seed1234.fits.gz".format(telescope)
        ),
        (0, 1, 2),
    )
    expected_map[expected_map == 0] = hp.UNSEEN
    np.testing.assert_allclose(output_map, expected_map)
