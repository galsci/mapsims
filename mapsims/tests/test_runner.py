import numpy as np
from astropy.tests.helper import assert_quantity_allclose
import healpy as hp

import pysm.units as u
from astropy.utils import data

import mapsims
import so_pysm_models

NSIDE = 16


def test_from_config():

    simulator = mapsims.from_config(
        data.get_pkg_data_filename("example_config.toml", package="mapsims")
    )
    output_map = simulator.execute(write_outputs=False)[simulator.channels[0]]

    expected_map = hp.read_map(
        data.get_pkg_data_filename("data/example_run.fits.gz"), (0, 1, 2)
    )
    assert_quantity_allclose(output_map, expected_map, rtol=1e-3)


def test_from_classes():

    noise = mapsims.SONoiseSimulator(
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
        seed=8974,
    )

    cmb = mapsims.SOPrecomputedCMB(
        iteration_num=0,
        nside=NSIDE,
        lensed=False,
        aberrated=False,
        has_polarization=True,
        cmb_set=0,
        cmb_dir="mapsims/tests/data",
        input_units="uK_CMB",
    )

    # CIB is only at NSIDE 4096, too much memory for testing
    # cib = so_pysm_models.WebSkyCIB(
    #    websky_version="0.3", nside=NSIDE, interpolation_kind="linear"
    # )

    simulator = mapsims.MapSim(
        channels=["SA_27"],
        nside=NSIDE,
        unit="uK_CMB",
        pysm_components_string="SO_d0",
        pysm_custom_components={"cmb": cmb},
        pysm_output_reference_frame="C",
        other_components={"noise": noise},
    )
    output_map = simulator.execute(write_outputs=False)[simulator.channels[0]]

    expected_map = hp.read_map(
        data.get_pkg_data_filename("data/example_run.fits.gz"), (0, 1, 2)
    )
    assert_quantity_allclose(output_map, expected_map, rtol=1e-3)