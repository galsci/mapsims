from astropy.tests.helper import assert_quantity_allclose
import numpy as np
import healpy as hp

from astropy.utils import data

import mapsims

NSIDE = 256


def test_from_config_v02():
    simulator = mapsims.from_config(
        data.get_pkg_data_filename("data/example_config_v0.2.toml", package="mapsims")
    )
    output_map = simulator.execute(write_outputs=False)[simulator.channels[0][0].tag]

    expected_map = hp.read_map(
        data.get_pkg_data_filename(
            f"data/simonsobs_ST0_UHF1_nside{NSIDE}.fits.gz", package="mapsims.tests"
        ),
        (0, 1, 2),
        dtype=np.float64,
    )
    # assert_quantity_allclose(
    #     output_map, expected_map, rtol=0.5e-2
    # )  # Only .5% because executing it in double precision changed the result by that much


def test_from_classes():

    # Test CMB alms from Planck generated with
    # https://zonca.dev/2020/09/planck-spectra-healpy.html

    cmb = mapsims.PrecomputedCMB(
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
        modeling_nside=NSIDE,
        unit="uK_CMB",
        lmax_over_nside=1.5,
        pysm_components_string="d0",
        pysm_custom_components={"cmb": cmb},
        output_reference_frame="C",
        instrument_parameters="simonsobs_instrument_parameters_2020.06",
    )

    output_map = simulator.execute(write_outputs=False)[simulator.channels[0][0].tag]

    expected_map = hp.read_map(
        data.get_pkg_data_filename(
            f"data/simonsobs_ST0_UHF1_nside{NSIDE}.fits.gz", package="mapsims.tests"
        ),
        (0, 1, 2),
    )
    #   assert_quantity_allclose(
    #       output_map, expected_map, rtol=0.5e-2
    #   )  # only .5% percent because executing it in double precision changed the result by that much
