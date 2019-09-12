from collections import namedtuple
import numpy as np
import astropy.units as u

import sotodlib.hardware

from . import SO_Noise_Calculator_Public_20180822 as so_noise

bands = ("LF1", "LF2", "MFF1", "MFF2", "MFS1", "MFS2", "UHF1", "UHF2")
frequencies = (27, 39, 93, 145, 93, 145, 225, 280)
hw = sotodlib.hardware.config.get_example()


class Channel:
    def __init__(self, tag, telescope, band, beam, bandpass):
        self.tag = tag
        self.telescope = telescope
        self.band = band
        self.beam = beam
        self.bandpass = bandpass

    @property
    def tag(self):
        return self.tag

    def get_beam(self):
        return self.beam

    def get_bandpass(self):
        return self.bandpass


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
            self.frequency = int(band)
            self.band = "{:03d}".format(self.frequency)
        except ValueError:
            self.frequency = frequencies[bands.index(band)]
            self.band = band

    @property
    def tag(self):
        return "_".join([self.telescope, self.band])

    def get_beam(self):
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
            else bands[frequencies.index(self.frequency)]
        )
        return hw.data["telescopes"][telescope_tag]["fwhm"][band] * u.arcmin

    def get_bandpass(self):
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
        return [Channel(channels, band) for band in bands]
    elif channels in ["all", "SO"]:
        return [
            Channel(telescope, band) for telescope in ["LA", "SA"] for band in bands
        ]
    else:
        if "," in channels:
            channels = channels.split(",")
        if isinstance(channels, str):
            channels = [channels]
        return [Channel(*ch.split("_")) for ch in channels]
