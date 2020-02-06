from mapsims import so_utils
import pytest


def test_single_ch():
    assert so_utils.parse_channels("SA_27")[0].tag == "SA_27"


@pytest.mark.parametrize("telescope", ["SA", "LA"])
def test_telescopes(telescope):
    assert len(so_utils.parse_channels(telescope)) == 8


def test_tube():
    channels = so_utils.parse_channels("ST3")
    assert len(channels) == 1
    assert isinstance(channels[0], tuple)
    assert channels[0][1].tag == "ST3_LF2"
    assert channels[0][0].tube == "ST3"
