from mapsims.channel_utils import parse_instrument_parameters
from mapsims.utils import DEFAULT_INSTRUMENT_PARAMETERS


def test_single_ch():
    ch = "ST3_LF2"
    assert parse_instrument_parameters(DEFAULT_INSTRUMENT_PARAMETERS, ch)[0].tag == ch


def test_telescopes():
    assert (
        len(parse_instrument_parameters(DEFAULT_INSTRUMENT_PARAMETERS, "telescope:SA"))
        == 8
    )
    assert (
        len(parse_instrument_parameters(DEFAULT_INSTRUMENT_PARAMETERS, "telescope:LA"))
        == 14
    )


def test_tube():
    channels = parse_instrument_parameters(DEFAULT_INSTRUMENT_PARAMETERS, "tube:ST3")
    assert len(channels) == 1
    assert isinstance(channels[0], tuple)
    assert channels[0][1].tag == "ST3_LF2"
    assert channels[0][0].tube == "ST3"
