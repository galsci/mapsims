from astropy.tests.helper import assert_quantity_allclose
import pytest

try:  # PySM >= 3.2.1
    import pysm3 as pysm
    import pysm3.units as u
except ImportError:
    import pysm
    import pysm.units as u

import mapsims

NSIDE = 16


@pytest.mark.skip(reason="cannot run without mpi")
def test_from_classes_custominstrument():

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

    # CIB is only at NSIDE 4096, too much memory for testing
    # cib = WebSkyCIB(
    #    websky_version="0.3", nside=NSIDE, interpolation_kind="linear"
    # )

    simulator = mapsims.MapSim(
        channels="100",
        nside=NSIDE,
        unit="uK_CMB",
        pysm_components_string="SO_d0",
        pysm_custom_components={"cmb": cmb},
        pysm_output_reference_frame="C",
        instrument_parameters="planck_deltabandpass",
    )
    output_map = simulator.execute(write_outputs=False)["100"]

    freq = 100.89 * u.GHz
    expected_map = cmb.get_emission(freq)
    fwhm = 9.682 * u.arcmin

    from mpi4py import MPI

    map_dist = pysm.MapDistribution(
        nside=NSIDE, smoothing_lmax=3 * NSIDE - 1, mpi_comm=MPI.COMM_WORLD
    )
    expected_map = pysm.mpi.mpi_smoothing(
        expected_map.to_value(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq)),
        fwhm,
        map_dist,
    )

    assert_quantity_allclose(output_map, expected_map, rtol=1e-3)
