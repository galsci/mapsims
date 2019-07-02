from collections import namedtuple
import numpy as np
import astropy.units as u

import sotodlib.hardware

from . import SO_Noise_Calculator_Public_20180822 as so_noise

bands = ("LF1", "LF2", "MFF1", "MFF2", "MFS1", "MFS2", "UHF1", "UHF2")
frequencies = (27, 39, 93, 145, 93, 145, 225, 280)
hw = sotodlib.hardware.config.get_example()


class Channel:
    def __init__(self, telescope, band):
        """Single Simons Observatory frequency channel

        Simple way of referencing a frequency band, this will be replaced
        in the future by a common metadata handling package

        Parameters
        ----------
        telescope : str
            LA or SA for Large and Small Aperture telescope
        band : str
            Band name, e.g. LF1
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


def get_bandpass(band):
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
        return int(band) * u.GHz, None
    except ValueError:
        properties = hw.data["bands"][band]
        return (
            np.linspace(properties["low"], properties["high"], 10) * u.GHz,
            np.ones(10, dtype=np.float32),
        )
