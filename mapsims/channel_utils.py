from astropy.utils import data
from astropy.table import QTable
import numpy as np
import astropy.units as u
import healpy as hp
from pathlib import Path
import logging
from typing import Optional

log = logging.getLogger("mapsims")


class Channel:
    @u.quantity_input
    def __init__(
        self,
        tag,
        telescope,
        band,
        tube,
        beam= None,
        center_frequency=None,
        custom_beam=None,
        nside=None,
        car_resol: u.arcmin = None,
        bandpass=None,
        **kwargs,
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
        custom_beam : np.array
            custom beam response, B_ell
        center_frequency : u.GHz
            center frequency of the channel, it is also necessary when a bandpass
            is provided
        nside : int
            Native HEALPix resolution parameter $N_{side}$
        car_resol: u.arcmin
            Native CAR pixelization resolution parameter in arcmin
        bandpass : (np.array, np.array)
            dimensionless frequency response of the channel, the weighting will
            be performed in power units, MJ/sr
        kwargs : other keyword arguments
            Any other keyword arguments is added as an attribute to the object
        """
        self.tag = tag
        self.telescope = telescope
        self.band = band
        self.beam = beam
        self.custom_beam = custom_beam
        self.tube = tube
        self.center_frequency = center_frequency
        self.nside = nside
        self.car_resol = car_resol
        self.bandpass = bandpass
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return "Channel " + self.tag


def parse_channels(filter="all", instrument_parameters=None):
    """Create a list of Channel objects from a HDF5 and an optional filter

    Reads a HDF5 file which contains the instruments parameters and
    parses it into a list of Channel objects, by default all channels
    are included.

    Parameters
    ----------
    instrument_parameters : str or Path
        See the instrument_parameters argument of MapSim
    filter : str or None
        See the channels argument of MapSim

    Returns
    -------
    channel_objects_list : list of Channel objects or list of tuples of Channel objects
        List of the selected Channel objects, in case of tubes, we return a list where
        each element is the tuple of the Channel objects in a tube. This is useful
        to simulate correlated noise.
    """

    assert (
        instrument_parameters is not None
    ), "Need to specify an instrument parameters file"

    if not isinstance(
        instrument_parameters, Path
    ) and not instrument_parameters.endswith("tbl"):
        instrument_parameters = data.get_pkg_data_filename(
            "data/{i}/{i}.tbl".format(i=instrument_parameters)
        )
    log.info("Parsing channels from %s", instrument_parameters)
    instrument_parameters = Path(instrument_parameters)
    # Need a valid filter_key to avoid errors below
    filter_key = "telescope"
    if filter != "all":
        if ":" not in filter:
            filter_values = filter
        else:
            filter_key, filter_values = filter.split(":")
        filter_values = filter_values.split(",")

    channel_objects_list = []
    table = QTable.read(str(instrument_parameters), format="ascii.ipac")

    for row in table:
        try:
            tag = row["tag"]
        except KeyError:
            tag = row["band"]

        if (
            filter == "all"
            or (tag in filter_values)
            or (row[filter_key] in filter_values)
        ):
            log.info("Channel %s", tag)
            try:
                telescope = row["telescope"]
            except KeyError:
                telescope = instrument_parameters.name.split(".")[0].split("_")[0]
            try:
                tube = row["tube"]
            except KeyError:
                tube = telescope

            try:
                beam_filename = instrument_parameters.parent / row["beam_file"]
                if beam_filename.suffix == ".fits":
                    beam = hp.read_cl(beam_filename)
                else:
                    beam = QTable.read(beam_filename, format="ascii.ipac")["B"]
                log.info("Beam from %s", beam_filename)
            except KeyError:
                log.info("No custom beam specified")

            try:
                bandpass_filename = instrument_parameters.parent / row["bandpass_file"]
                bandpass = QTable.read(bandpass_filename, format="ascii.ipac")

                log.info("Bandpass from %s", bandpass_filename)
            except KeyError:
                try:
                    low = row["center_frequency"] - row["bandwidth"] / 2
                    bandpass = {
                        "bandpass_frequency": np.linspace(
                            low, low + row["bandwidth"], 10
                        ),
                        "bandpass_weight": np.ones(10, dtype=np.float),
                    }
                    log.info(
                        "Top-hat bandpass based on center_frequency and bandwidth [%.2g, %.2g] %s",
                        low.value,
                        (low + row["bandwidth"]).value,
                        low.unit,
                    )
                except KeyError:
                    bandpass = {
                        "bandpass_frequency": row["center_frequency"],
                        "bandpass_weight": np.ones(1),
                    }
                    log.info("Delta-frequency bandpass on center frequency")
            # this loops through all column names so should pickup also
            # nside and car_resol
            other_metadata = {
                name: row[name]
                for name in row.colnames
                if name not in ["band", "tag", "fwhm", "telescope", "tube"]
            }

            try:
                other_metadata["custom_beam"] = beam
            except UnboundLocalError:
                pass

            channel_objects_list.append(
                Channel(
                    tag=tag,
                    band=row["band"],
                    bandpass=(
                        bandpass["bandpass_frequency"],
                        bandpass["bandpass_weight"],
                    ),
                    beam=row.get("fwhm", None),
                    telescope=telescope,
                    tube=tube,
                    **other_metadata,
                )
            )
    # just for tubes, return tuples of channel pairs
    if filter_key == "tube":
        out = []
        for tube in filter_values:
            out.append(tuple([ch for ch in channel_objects_list if ch.tube == tube]))
        channel_objects_list = out

    return channel_objects_list
