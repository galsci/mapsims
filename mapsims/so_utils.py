try:
    import h5py
except ImportError:
    h5py = None
from astropy.utils import data
import numpy as np
import astropy.units as u
from pathlib import Path

import sotodlib.hardware

from . import SO_Noise_Calculator_Public_20180822 as so_noise

bands = ("LF1", "LF2", "MFF1", "MFF2", "MFS1", "MFS2", "UHF1", "UHF2")
frequencies = (27, 39, 93, 145, 93, 145, 225, 280)
hw = sotodlib.hardware.config.get_example()


class Channel:
    @u.quantity_input
    def __init__(
        self,
        tag,
        telescope,
        band,
        beam: u.arcmin,
        center_frequency: u.GHz,
        bandpass=None,
    ):
        """Base class of a channel

        Each channel will be used to produce an output map

        Parameters
        ----------

        tag : str
            channel identifier
        telescope : str
            telescope name
        band : str
            identifier of the frequency band, useful for multiple channels
            with different beams but same frequency response
        beam : u.arcmin
            full-width-half-max of the beam, assumed gaussian
        center_frequency : u.GHz
            center frequency of the channel, it is also necessary when a bandpass
            is provided
        bandpass : (np.array, np.array)
            dimensionless frequency response of the channel, the weighting will
            be performed in power units, MJ/sr
        """
        self.tag = tag
        self.telescope = telescope
        self.band = band
        self.beam = beam
        self.center_frequency = center_frequency
        self.bandpass = bandpass

    def __repr__(self):
        return "Channel " + self.tag


class SOChannel(Channel):
    def __init__(self, telescope, band):
        """Single Simons Observatory frequency channel

        Simple way of referencing a frequency band, this will be replaced
        in the future by a common metadata handling package

        Parameters
        ----------
        telescope : str
            LA or SA for Large and Small Aperture telescope
        band : str
            Band name, e.g. LF1 or frequency, e.g. 93
        """
        self.telescope = telescope
        try:
            self.center_frequency = int(band) * u.GHz
            self.band = "{}".format(int(self.center_frequency.value))
        except ValueError:
            self.center_frequency = frequencies[bands.index(band)] * u.GHz
            self.band = band

    @property
    def tag(self):
        return "_".join([self.telescope, self.band])

    @property
    def beam(self):
        """Returns the beam in arcminutes for a band

        Returns
        -------
        beam : astropy.units.Quantity
            Full width half max in arcmin
        """
        telescope_tag = {"SA": "SAT1", "LA": "LAT"}[self.telescope]

        band = (
            self.band
            if self.band in bands
            else bands[frequencies.index(int(self.center_frequency.value))]
        )
        return hw.data["telescopes"][telescope_tag]["fwhm"][band] * u.arcmin

    @property
    def bandpass(self):
        """Returns tophat bandpass

        10 points between minimun and maximum with equal weights

        Returns
        -------
        frequency : astropy.units.Quantity
            array of 10 frequency points equally spaced between min and max bandpass
        weights : np.array
            array of ones
        """
        try:
            return int(self.band) * u.GHz, None
        except ValueError:
            properties = hw.data["bands"][self.band]
            return (
                np.linspace(properties["low"], properties["high"], 10) * u.GHz,
                np.ones(10, dtype=np.float32),
            )


def parse_channels(channels):
    """Parse a reference string into Channel objects

    For example turns "LA" in a list of all LA Channels,
    reference all channels either with "all" or "SO".

    Parameters
    ----------
    channels : str
        reference string to one or more channels, supported
        values are LA,SA,all,SO and comma separated channel names
        e.g. LA_LF1,SA_LF1

    Returns
    -------
    channels : list of Channel objects
        list of Channel objects
    """

    if channels in ["LA", "SA"]:
        return [SOChannel(channels, band) for band in bands]
    elif channels in ["all", "SO"]:
        return [
            SOChannel(telescope, band) for telescope in ["LA", "SA"] for band in bands
        ]
    else:
        if "," in channels:
            channels = channels.split(",")
        if isinstance(channels, str):
            channels = [channels]
        return [SOChannel(*ch.split("_")) for ch in channels]


def parse_instrument_parameters(instrument_parameters, channels="all"):
    if not isinstance(instrument_parameters, Path) and not instrument_parameters.endswith("h5"):
        instrument_parameters = data.get_pkg_data_filename(
            "data/{}.h5".format(instrument_parameters)
        )
    instrument_parameters = Path(instrument_parameters)
    if h5py is None:
        raise ImportError("h5py is needed to parse instrument parameter files")
    channel_objects_list = []
    with h5py.File(instrument_parameters) as f:
        if channels == "all":
            channels = f.keys()
        if isinstance(channels, str):
            channels = [channels]
        for ch in channels:
            channel_objects_list.append(
                Channel(
                    tag=ch,
                    band=f[ch].attrs["band"],
                    beam=f[ch].attrs["fwhm_arcmin"] * u.arcmin,
                    center_frequency=f[ch].attrs["center_frequency_GHz"] * u.GHz,
                    telescope=instrument_parameters.name.split(".")[0].split("_")[0],
                    bandpass=(
                        f[ch].get(
                            "bandpass_frequency_GHz",
                            default=f[ch].attrs["center_frequency_GHz"],
                        )
                        * u.GHz,
                        f[ch].get("bandpass_weight"),
                    ),
                )
            )
    return channel_objects_list
