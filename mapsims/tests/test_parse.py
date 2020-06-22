import astropy.units as u
import pytest
import h5py
from mapsims.channel_utils import parse_channels


@pytest.fixture
def create_test_instrument_parameters(tmp_path):
    tmp_filename = tmp_path / "test_params.h5"
    with h5py.File(tmp_filename, "w") as instrument_parameters:
        for tag in ["ch_1", "ch_2"]:
            ch = instrument_parameters.create_group(tag)
            ch.attrs["band"] = tag
            ch.attrs["fwhm_arcmin"] = 12
            ch.attrs["center_frequency_GHz"] = 70
    return tmp_filename


def test_parse_instrument_parameters_all(create_test_instrument_parameters):

    tmp_filename = create_test_instrument_parameters
    channels = parse_channels(instrument_parameters=tmp_filename)
    assert len(channels) == 2
    assert channels[0].tag == "ch_1"
    assert channels[1].tag == "ch_2"
    for each in channels:
        assert each.beam == 12 * u.arcmin
        assert each.center_frequency == 70 * u.GHz


def test_parse_instrument_parameters_subset(create_test_instrument_parameters):

    tmp_filename = create_test_instrument_parameters
    channels = parse_channels(instrument_parameters=tmp_filename, filter="ch_1")
    assert len(channels) == 1
    assert channels[0].tag == "ch_1"
    for each in channels:
        assert each.beam == 12 * u.arcmin
        assert each.center_frequency == 70 * u.GHz
