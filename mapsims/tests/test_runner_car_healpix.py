from astropy.tests.helper import assert_quantity_allclose
import healpy as hp

from astropy.utils import data
import pysm3.units as u

import mapsims

NSIDE = 16


def test_from_classes_car_healpix(tmp_path):

    output_folder = tmp_path / "write_car_healpix"

    ch = "tube:ST0"

    simulator = mapsims.MapSim(
        channels=ch,
        nside=NSIDE,
        unit="uK_CMB",
        pysm_components_string="d1",
        output_reference_frame="G",
        car=True,
        car_resolution=1 * u.deg,
        healpix=True,
        output_folder=output_folder,
    )

    first_ch = simulator.channels[0][0].tag
    output_maps = simulator.execute(write_outputs=False)[first_ch]
    assert len(output_maps) == 2
    assert output_maps[0].shape == (3, hp.nside2npix(NSIDE))
    assert output_maps[1].shape[0] == 3
    assert output_maps[1].ndim == 3

    filenames = simulator.execute(write_outputs=True)

    healpix_map_fromdisk = hp.read_map(
        output_folder / filenames[first_ch][0], (0, 1, 2)
    )
    assert_quantity_allclose(healpix_map_fromdisk, output_maps[0], rtol=1e-6)
