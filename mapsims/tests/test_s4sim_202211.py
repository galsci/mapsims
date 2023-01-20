import os
import healpy as hp
from numpy.testing import assert_allclose

from ..runner import command_line_script


def test_s4sim_202222_ame_high():
    command_line_script(
        [
            "mapsims/tests/data/common.toml",
            "mapsims/tests/data/ame_high.toml",
            "--verbose",
            "--channels",
            "LFL1",
            "--nside",
            "16",
        ]
    )

    filename = "cmbs4_ame_high_uKCMB_LAT-LFL1_nside16_0000.fits"
    output_map = hp.read_map(filename, (0, 1, 2))

    expected_map = hp.read_map(f"mapsims/tests/data/{filename}.gz", (0, 1, 2))

    assert_allclose(expected_map, output_map, rtol=1e-4)
    os.remove(filename)
