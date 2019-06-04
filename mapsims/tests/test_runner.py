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

    dust = so_pysm_models.GaussianDust(
        nside=NSIDE,
        has_polarization=True,
        TT_amplitude=350.,
        Toffset=18.,
        EE_amplitude=100.,
        rTE=0.35,
        EtoB=0.5,
        alpha=-0.42,
        beta=1.53,
        temp=19.6,
        nu_0=353,
        seed=20001,
    )

    sync = so_pysm_models.GaussianSynchrotron(
        nside=NSIDE,
        has_polarization=True,
        TT_amplitude=20.0,
        Toffset=72.,
        EE_amplitude=4.3,
        rTE=0.35,
        EtoB=0.5,
        alpha=-1.0,
        beta=-3.1,
        curv=0.,
        nu_0=23.,
        seed=30001,
    )

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

    simulator = mapsims.MapSim(
        channels=["SA_27"],
        nside=NSIDE,
        unit="uK_CMB",
        pysm_components_string="a1",
        pysm_custom_components={"dust": dust, "synchrotron": sync, "cmb": cmb},
        pysm_output_reference_frame="G",
        other_components={"noise": noise},
    )
    output_map = simulator.execute(write_outputs=False)[simulator.channels[0]]

    expected_map = hp.read_map(
        data.get_pkg_data_filename("data/example_run.fits.gz"), (0, 1, 2)
    )
    assert_quantity_allclose(output_map, expected_map, rtol=1e-3)
