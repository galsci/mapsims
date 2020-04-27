import numpy as np
import healpy as hp

import pytest
from astropy.tests.helper import assert_quantity_allclose

from astropy.utils import data
import mapsims
import pysm.units as u
from mapsims import so_utils

NSIDE = 16

def test_freq_order():
    freqs = np.unique(so_utils.frequencies)
    ifreqs = (27, 39, 93, 145, 225, 280)
    for f,f2 in zip(freqs,ifreqs):
        assert f==f2

@pytest.mark.parametrize("telescope", ["SA", "LA"])
def test_noise_simulator(telescope):

    seed = 1234 - 200
    if telescope == "SA":
        seed -= 1000

    simulator = mapsims.SONoiseSimulator(
        telescopes=["LA", "SA"],
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
        LA_noise_model="SOLatV3",
        SA_years=365 * 5 / 365.25,
        SA_number_LF=0.2,
        SA_number_MF=1.8,
        SA_number_UHF=1,
        SA_one_over_f_mode="optimistic",
        num=seed,
    )

    output_map = simulator.simulate(mapsims.SOChannel(telescope, "MFF1")) * u.uK_CMB
    expected_map = hp.read_map(
        data.get_pkg_data_filename(
            "data/noise_{}_uKCMB_classical_nside16_channel2_seed1234.fits.gz".format(
                telescope
            )
        ),
        (0, 1, 2),
    )
    expected_map[expected_map == 0] = hp.UNSEEN
    expected_map <<= u.uK_CMB
    assert_quantity_allclose(output_map, expected_map)


"""
Test cases

pixelizations = ['car','healpix']
tubes = ['LT0','ST2']
cov = [True,False]
homogenous = [True, False]
cache_hitmaps = [True,False]



"""
