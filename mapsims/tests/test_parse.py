import numpy as np
import astropy.units as u
import pytest
from astropy.table import QTable
from mapsims.channel_utils import parse_channels


@pytest.fixture
def create_test_instrument_parameters(tmp_path):
    tmp_filename = tmp_path / "test_params.tbl"
    instrument_model = []
    bandpass = {
        "bandpass_frequency": np.arange(10, 20) * u.GHz,
        "bandpass_weight": np.ones(10),
    }
    for tag in ["ch_1", "ch_2"]:
        ch = {}
        ch["band"] = tag
        ch["fwhm"] = 12 * u.arcmin
        ch["center_frequency"] = 70 * u.GHz
        ch["telescope"] = "telescope_" + tag[-1]
        instrument_model.append(ch)
        QTable(bandpass).write(tmp_path / f"bandpass_{tag}.tbl", format="ascii.ipac")
    QTable(instrument_model).write(tmp_filename, format="ascii.ipac")

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
        assert len(each.bandpass[0]) == 10
        assert each.bandpass[0].min() == 10 * u.GHz
        assert sum(each.bandpass[1]) == 10


def test_parse_instrument_parameters_subset(create_test_instrument_parameters):

    tmp_filename = create_test_instrument_parameters
    channels = parse_channels(instrument_parameters=tmp_filename, filter="ch_1")
    assert len(channels) == 1
    assert channels[0].tag == "ch_1"
    for each in channels:
        assert each.beam == 12 * u.arcmin
        assert each.center_frequency == 70 * u.GHz


def test_parse_instrument_parameters_filtertube(create_test_instrument_parameters):

    tmp_filename = create_test_instrument_parameters
    channels = parse_channels(
        instrument_parameters=tmp_filename, filter="telescope:telescope_2"
    )
    assert len(channels) == 1
    assert channels[0].tag == "ch_2"
    for each in channels:
        assert each.beam == 12 * u.arcmin
        assert each.center_frequency == 70 * u.GHz
