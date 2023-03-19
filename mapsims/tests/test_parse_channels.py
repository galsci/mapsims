from mapsims.channel_utils import parse_channels
from functools import partial

# Use currying to set the default instrument parameters file for
# all the following tests

parse_channels = partial(
    parse_channels, instrument_parameters="simonsobs_instrument_parameters_2020.06"
)


def test_single_ch():
    ch = "ST3_LF2"
    assert parse_channels(ch)[0].tag == ch


def test_telescopes():
    assert len(parse_channels("telescope:SA")) == 8
    assert len(parse_channels("telescope:LA")) == 14


def test_tube():
    channels = parse_channels("tube:ST3")
    assert len(channels) == 1
    assert isinstance(channels[0], tuple)
    assert channels[0][1].tag == "ST3_LF2"
    assert channels[0][0].tube == "ST3"
