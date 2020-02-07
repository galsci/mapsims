import os.path
import importlib
import numpy as np

import pysm
from pysm import units as u
import toml

import healpy as hp

from so_pysm_models import get_so_models

try:
    from mpi4py import MPI

    COMM_WORLD = MPI.COMM_WORLD
except ImportError:
    COMM_WORLD = None

import warnings

from . import so_utils
from . import Channel

PYSM_COMPONENTS = {
    comp[0]: comp for comp in ["synchrotron", "dust", "freefree", "cmb", "ame"]
}
default_output_filename_template = (
    "simonsobs_{telescope}_{band}_nside{nside}_{split}_of_{nsplits}.fits"
)


def function_accepts_argument(func, arg):
    """Check if a function or class accepts argument arg

    Parameters
    ----------

    func : Python function or Class
        Input Python function or Class to check
    arg : str
        keyword or positional argument to check

    Returns
    -------

    accepts_argument : bool
        True if function/class constructor accepts argument
    """
    if not hasattr(func, "__code__"):
        func = func.__init__
    return arg in func.__code__.co_varnames


def command_line_script(args=None):

    import argparse

    parser = argparse.ArgumentParser(
        description="Execute map based simulations for Simons Observatory"
    )
    parser.add_argument("config", type=str, help="Configuration file", nargs="+")
    parser.add_argument("--nside", type=int, required=False, help="NSIDE")
    parser.add_argument(
        "--num",
        type=int,
        required=False,
        help="Simulation number, generally used as seed",
    )
    parser.add_argument(
        "--channels",
        type=str,
        help="Channels e.g. all, SA, LA, LA_27, ST2",
        required=False,
    )
    res = parser.parse_args(args)
    override = {
        key: getattr(res, key)
        for key in ["nside", "channels", "num"]
        if getattr(res, key) is not None
    }

    simulator = from_config(res.config, override=override)
    simulator.execute(write_outputs=True)


def import_class_from_string(class_string):
    module_name, class_name = class_string.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def from_config(config_file, override=None):
    if isinstance(config_file, str):
        config_file = [config_file]

    config = toml.load(config_file)
    if override is not None:
        config.update(override)

    pysm_components_string = None
    pysm_output_reference_frame = None

    components = {}
    for component_type in ["pysm_components", "other_components"]:
        components[component_type] = {}
        if component_type in config:
            component_type_config = config[component_type]
            if component_type == "pysm_components":
                pysm_components_string = component_type_config.pop(
                    "pysm_components_string", None
                )
                pysm_output_reference_frame = component_type_config.pop(
                    "pysm_output_reference_frame", None
                )
            for comp_name in component_type_config:
                comp_config = component_type_config[comp_name]
                comp_class = import_class_from_string(comp_config.pop("class"))
                if (
                    function_accepts_argument(comp_class, "num")
                    and "num" in config
                    and "num" not in comp_config
                ):
                    # If a component has an argument "num" and we provide a configuration
                    # "num" to MapSims, we pass it to all the class.
                    # it can be overridden by the actual component config
                    # This is used for example by `SOStandalonePrecomputedCMB`
                    comp_config["num"] = config["num"]
                components[component_type][comp_name] = comp_class(
                    nside=config["nside"], **comp_config
                )

    map_sim = MapSim(
        channels=config["channels"],
        nside=int(config["nside"]),
        num=config["num"],
        nsplits=config.get("nsplits", 1),
        unit=config["unit"],
        tag=config["tag"],
        output_folder=config.get("output_folder", "output"),
        output_filename_template=config.get(
            "output_filename_template", default_output_filename_template
        ),
        pysm_components_string=pysm_components_string,
        pysm_custom_components=components["pysm_components"],
        pysm_output_reference_frame=pysm_output_reference_frame,
        other_components=components["other_components"],
        instrument_parameters=config.get("instrument_parameters", None),
    )
    return map_sim


class MapSim:
    def __init__(
        self,
        channels,
        nside,
        num=0,
        nsplits=1,
        unit="uK_CMB",
        output_folder="output",
        tag="mapsim",
        output_filename_template=default_output_filename_template,
        pysm_components_string=None,
        pysm_output_reference_frame="C",
        pysm_custom_components=None,
        other_components=None,
        instrument_parameters=None,
    ):
        """Run map based simulations

        MapSim executes PySM for each of the input channels with a sky defined
        by default PySM components in `pysm_components_string` and custom components in
        `pysm_custom_components` and rotates in Alm space to the reference frame `pysm_output_reference_frame`.
        Then for each of the channels specified, smoothes the map with the channel beam
        and finally adds the map generated by `other_components`, for example noise maps, and writes
        the outputs to disk.

        Parameters
        ----------

        channels : str
            all/SO for all channels, LA for all Large Aperture channels, SA for Small,
            otherwise a single channel label, e.g. LA_27 or a list of channel labels,
            or "027" for "LA_27" and "SA_27"
            to simulate a tube, leave channels None and set tube instead
        tube : str
            tube to simulate, it is necessary for noise simulations with hitmaps v0.2
            currently only 1 tube at a time is supported
            see mapsims.so_utils.tubes for the available tubes
        nside : int
            output HEALPix Nside
        unit : str
            Unit of output maps
        output_folder : str
            Relative or absolute path to output folder, string template with {nside} and {tag} fields
        num : int
            Realization number, generally used as seed, default is 0, automatically padded to 4 digits
        nsplits : int
            Number of noise splits, see the documentation of :py:class:`SONoiseSimulator`
        tag : str
            String to describe the current simulation, for example its content, which is used into
            string interpolation for `output_folder` and `output_filename_template`
        output_filename_template : str
            String template with {telescope} {channel} {nside} {tag} fields
        pysm_components_string : str
            Comma separated string of PySM components, i.e. "s1,d4,a2"
        pysm_output_reference_frame : str
            The output of PySM is in Galactic coordinates, rotate to C for Equatorial or E for Ecliptic,
            set to None to apply no rotation
        pysm_custom_components : dict
            Dictionary of other components executed through PySM
        other_components : dict
            Dictionary of component name, component class pairs, the output of these are **not** rotated,
            they should already be in the same reference frame specified in pysm_output_reference_frame.
        instrument_parameters : HDF5 file path or str
            Instrument parameters in HDF5 format, each channel tag is a group, each group has attributes
            band, center_frequency_GHz, fwhm_arcmin, bandpass_frequency_GHz, bandpass_weight


        """

        self.tube = None
        if instrument_parameters is None:
            self.channels = so_utils.parse_channels(channels)
            if isinstance(self.channels[0], tuple):
                self.tube = self.channels[0][0].tube
        else:
            self.channels = so_utils.parse_instrument_parameters(
                instrument_parameters, channels
            )

        self.nside = nside
        self.unit = unit
        self.num = num
        self.nsplits = nsplits
        self.pysm_components_string = pysm_components_string
        self.pysm_custom_components = pysm_custom_components
        self.run_pysm = not (
            (pysm_components_string is None)
            and (pysm_custom_components is None or len(pysm_custom_components) == 0)
        )
        self.other_components = other_components
        self.tag = tag
        self.output_folder = output_folder.format(
            nside=self.nside, tag=self.tag, num=self.num
        )
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.output_filename_template = output_filename_template
        self.rot = None
        self.pysm_output_reference_frame = pysm_output_reference_frame

    def execute(self, write_outputs=False):
        """Run map simulations

        Execute simulations for all channels and write to disk the maps,
        unless `write_outputs` is False, then return them.
        """

        if self.run_pysm:
            sky_config = []
            preset_strings = []
            if self.pysm_components_string is not None:
                for model in self.pysm_components_string.split(","):
                    if model.startswith("SO"):
                        sky_config.append(get_so_models(model, self.nside))
                    else:
                        preset_strings.append(model)

            if len(preset_strings) > 0:
                input_reference_frame = "G"
                assert (
                    len(sky_config) == 0
                ), "Cannot mix PySM and SO models, they are defined in G and C frames"
            else:
                input_reference_frame = "C"

            self.pysm_sky = pysm.Sky(
                nside=self.nside,
                preset_strings=preset_strings,
                component_objects=sky_config,
                output_unit=u.Unit(self.unit),
            )

            if self.pysm_custom_components is not None:
                for comp_name, comp in self.pysm_custom_components.items():
                    self.pysm_sky.components.append(comp)

        if not write_outputs:
            output = {}

        # ch can be single channel or tuple of 2 channels (tube dichroic)
        for ch in self.channels:
            if not isinstance(ch, tuple):
                ch = [ch]
            output_map = hp.ma(
                np.zeros((len(ch), 3, hp.nside2npix(self.nside)), dtype=np.float64)
            )
            if self.run_pysm:
                for each, channel_map in zip(ch, output_map):
                    bandpass_integrated_map = self.pysm_sky.get_emission(
                        *each.bandpass
                    ).value
                    beam_width_arcmin = each.beam
                    # smoothing and coordinate rotation with 1 spherical harmonics transform
                    channel_map += hp.ma(
                        pysm.apply_smoothing_and_coord_transform(
                            bandpass_integrated_map,
                            fwhm=beam_width_arcmin,
                            lmax=3 * self.nside - 1,
                            rot=hp.Rotator(
                                coord=(
                                    input_reference_frame,
                                    self.pysm_output_reference_frame,
                                )
                            ),
                            map_dist=pysm.MapDistribution(
                                nside=self.nside,
                                smoothing_lmax=3 * self.nside - 1,
                                mpi_comm=COMM_WORLD,
                            ),
                        )
                    )

            output_map = output_map.reshape((len(ch), 1, 3, -1))

            if self.other_components is not None:
                for comp in self.other_components.values():
                    kwargs = dict(tube=self.tube, output_units=self.unit)
                    if function_accepts_argument(comp.simulate, "nsplits"):
                        kwargs["nsplits"] = self.nsplits
                    component_map = hp.ma(comp.simulate(ch[0], **kwargs))
                    output_map = output_map + component_map

            for each, channel_map in zip(ch, output_map):
                if write_outputs:
                    for split, each_split_channel_map in enumerate(channel_map):
                        filename = self.output_filename_template.format(
                            telescope=each.telescope
                            if self.tube is None
                            else self.tube,
                            band=each.band,
                            nside=self.nside,
                            tag=self.tag,
                            num=self.num,
                            nsplits=self.nsplits,
                            split=split,
                        )
                        warnings.warn("Writing output map " + filename)
                        hp.write_map(
                            os.path.join(self.output_folder, filename),
                            each_split_channel_map,
                            coord=self.pysm_output_reference_frame,
                            column_units=self.unit,
                            overwrite=True,
                        )
                else:
                    if self.nsplits == 1:
                        channel_map = channel_map[0]
                    output[each.tag] = channel_map.filled()
        if not write_outputs:
            return output
