from astropy.tests.helper import assert_quantity_allclose
import healpy as hp

from astropy.utils import data
import pysm3.units as u

import mapsims

NSIDE = 16


def test_from_classes_car_healpix():

    simulator = mapsims.MapSim(
        channels="tube:ST0",
        nside=NSIDE,
        unit="uK_CMB",
        pysm_components_string="d1",
        pysm_output_reference_frame="G",
        car=True,
        car_resolution=1 * u.deg,
        healpix=True,
    )

    output_maps = simulator.execute(write_outputs=False)[simulator.channels[0][0].tag]
    assert len(output_maps) == 2
    assert output_maps[0].shape == (3, hp.nside2npix(NSIDE))
    assert output_maps[1].shape[0] == 3
    assert output_maps[1].ndim == 3
