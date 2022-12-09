import healpy as hp
from numpy.testing import assert_allclose

from ..runner import command_line_script

def test_s4sim_202222_ame_high():
    command_line_script(["mapsims/tests/data/common.toml", "mapsims/tests/data/ame_high.toml",
        "--verbose", "--channels", 'LFL1', "--nside", "16"])

    output_map = hp.read_map("cmbs4_ame_high_uKCMB_LAT-LFL1_nside16_0000.fits", (0,1,2))

    expected_map = hp.read_map("mapsims/tests/data/cmbs4_ame_high_uKCMB_LAT-LFL1_nside16_0000.fits.gz", (0,1,2))

    assert_allclose(expected_map, output_map)
