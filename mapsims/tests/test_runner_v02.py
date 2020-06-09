from astropy.tests.helper import assert_quantity_allclose
import healpy as hp

from astropy.utils import data

import mapsims

NSIDE = 16


def test_from_config_v02():

    simulator = mapsims.from_config(
        data.get_pkg_data_filename("data/example_config_v0.2.toml", package="mapsims")
    )
    output_map = simulator.execute(write_outputs=False)[simulator.channels[0][0].tag]

    expected_map = hp.read_map(
        data.get_pkg_data_filename("data/simonsobs_ST0_UHF1_nside16.fits.gz"), (0, 1, 2)
    )
    assert_quantity_allclose(output_map, expected_map, rtol=1e-6)


def test_from_classes():

    noise = mapsims.SONoiseSimulator(
        nside=NSIDE,
        return_uK_CMB=True,
        sensitivity_mode="baseline",
        apply_beam_correction=False,
        apply_kludge_correction=True,
        SA_one_over_f_mode="pessimistic",
    )

    cmb = mapsims.SOPrecomputedCMB(
        num=0,
        nside=NSIDE,
        lensed=False,
        aberrated=False,
        has_polarization=True,
        cmb_set=0,
        cmb_dir="mapsims/tests/data",
        input_units="uK_CMB",
    )

    simulator = mapsims.MapSim(
        channels="tube:ST0",
        nside=NSIDE,
        unit="uK_CMB",
        pysm_components_string="SO_d0",
        pysm_custom_components={"cmb": cmb},
        pysm_output_reference_frame="C",
        other_components={"noise": noise},
    )

    output_map = simulator.execute(write_outputs=False)[simulator.channels[0][0].tag]

    expected_map = hp.read_map(
        data.get_pkg_data_filename("data/simonsobs_ST0_UHF1_nside16.fits.gz"), (0, 1, 2)
    )
    assert_quantity_allclose(output_map, expected_map, rtol=1e-6)
