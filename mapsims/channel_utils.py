try:
    import h5py
except ImportError:
    h5py = None
from astropy.utils import data
import numpy as np
import astropy.units as u
from pathlib import Path
from .utils import DEFAULT_INSTRUMENT_PARAMETERS


class Channel:
    @u.quantity_input
    def __init__(
        self,
        tag,
        telescope,
        band,
        tube,
        beam: u.arcmin,
        center_frequency: u.GHz,
        bandpass=None,
        **kwargs
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
        kwargs : other keyword arguments
            Any other keyword arguments is added as an attribute to the object
        """
        self.tag = tag
        self.telescope = telescope
        self.band = band
        self.beam = beam
        self.tube = tube
        self.center_frequency = center_frequency
        self.bandpass = bandpass
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return "Channel " + self.tag


def parse_channels(filter="all", instrument_parameters=DEFAULT_INSTRUMENT_PARAMETERS):
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
    if not isinstance(
        instrument_parameters, Path
    ) and not instrument_parameters.endswith("h5"):
        instrument_parameters = data.get_pkg_data_filename(
            "data/{}.h5".format(instrument_parameters)
        )
    instrument_parameters = Path(instrument_parameters)
    if h5py is None:
        raise ImportError("h5py is needed to parse instrument parameter files")

    # Need a valid filter_key to avoid errors below
    filter_key = "fwhm_arcmin"
    if filter != "all":
        if ":" not in filter:
            filter_values = filter
        else:
            filter_key, filter_values = filter.split(":")
        filter_values = filter_values.split(",")

    channel_objects_list = []
    with h5py.File(instrument_parameters, "r") as f:

        for ch in f.keys():
            if filter == "all" or (
                ch in filter_values or f[ch].attrs[filter_key] in filter_values
            ):
                properties = dict(f[ch].attrs)
                if "telescope" not in properties:
                    properties["telescope"] = instrument_parameters.name.split(".")[
                        0
                    ].split("_")[0]
                properties["beam"] = properties.pop("fwhm_arcmin") * u.arcmin
                properties["center_frequency"] = (
                    properties.pop("center_frequency_GHz") * u.GHz
                )
                if "tube" not in properties:
                    properties["tube"] = properties["telescope"]
                channel_objects_list.append(
                    Channel(
                        tag=ch,
                        bandpass=(
                            np.array(
                                f[ch].get(
                                    "bandpass_frequency_GHz",
                                    default=properties["center_frequency"].value,
                                )
                            )
                            * u.GHz,
                            np.array(f[ch].get("bandpass_weight")),
                        ),
                        **properties
                    )
                )
    if filter_key == "tube":
        out = []
        for tube in filter_values:
            out.append(tuple([ch for ch in channel_objects_list if ch.tube == tube]))
        channel_objects_list = out

    return channel_objects_list
