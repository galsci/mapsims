import importlib
import os
import os.path
from astropy.table import Table
from astropy.utils import data
import healpy as hp
import numpy as np

try:  # PySM >= 3.2.1
    import pysm3.units as u
    import pysm3 as pysm
except ImportError:
    import pysm.units as u
    import pysm
import toml

# pixell is optional and needed when CAR simulations are requested
try:
    import pixell
    import pixell.curvedsky
    import pixell.powspec
except:
    pixell = None

from so_pysm_models import get_so_models
from .utils import DEFAULT_INSTRUMENT_PARAMETERS, merge_dict

import socket
on_cori_login = socket.gethostname().startswith("cori")

try:
    if "DISABLE_MPI" in os.environ or on_cori_login:
        raise ImportError
    from mpi4py import MPI

    COMM_WORLD = MPI.COMM_WORLD
except ImportError:
    COMM_WORLD = None

import warnings

from .channel_utils import parse_channels

PYSM_COMPONENTS = {
    comp[0]: comp for comp in ["synchrotron", "dust", "freefree", "cmb", "ame"]
}
default_output_filename_template = (
    "simonsobs_{tag}_{telescope}_{band}_nside{nside}_{split}_of_{nsplits}.fits"
)


def get_default_so_resolution(ch, field="NSIDE"):
    "Load the default Simons Observatory resolution"

    default_resolution = Table.read(
        data.get_pkg_data_filename("data/so_default_resolution.csv")
    )
    default_resolution.add_index("channel")
    first_ch = ch if not isinstance(ch, tuple) else ch[0]
    output = default_resolution.loc[first_ch.telescope + "_" + str(first_ch.band)][
        field
    ]
    if field == "CAR_resol":
        output *= u.arcmin
    return output


def get_map_shape(ch, nside=None, car_resolution=None, car=False):
    """Get map shape (and WCS for CAR) for Simons Observatory channels

    If N_side or car_resolution is None, get the default value
    from: `mapsims/data/default_resolution.csv`

    Parameters
    ----------
    ch : string
        Channel tag, e.g. SA_LF2
    nside : int
        Desired healpix N_side
    car_resolution : astropy.Quantity
        CAR pixels resolution with angle unit
    car : bool
        Set to True for CAR, False for HEALPix

    Returns
    -------
    nside : int
        N_side, either return the input or default
        None for CAR
    shape : tuple of int
        (npix,) for HEALPix, (Nx, Ny) for CAR
    wcs : astropy.WCS
        CAR map WCS, None for HEALPix
    """
    if car:
        if car_resolution is None:
            car_resolution = get_default_so_resolution(ch, field="CAR_resol")
        shape, wcs = pixell.enmap.fullsky_geometry(
            res=car_resolution.to_value(u.radian)
        )
        nside = None
    else:
        if nside is None:
            nside = get_default_so_resolution(ch)
        shape = (hp.nside2npix(nside),)
        wcs = None
    return nside, shape, wcs


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
        "--nsplits", type=int, required=False, help="Number of noise splits"
    )
    parser.add_argument(
        "--channels",
        type=str,
        help="Channels e.g. all, 'LT1_UHF1,LT0_UHF1', 'tube:LT1', see docstring of MapSim",
        required=False,
    )
    res = parser.parse_args(args)
    override = {
        key: getattr(res, key)
        for key in ["nside", "channels", "num", "nsplits"]
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

    config = toml.load(config_file[0])
    for conf in config_file[1:]:
        merge_dict(config, toml.load(conf))

    if override is not None:
        config.update(override)

    pysm_components_string = None
    pysm_output_reference_frame = None

    nside = config.get("nside", None)
    car = config.get("car", False)
    channels = parse_channels(config["channels"], config["instrument_parameters"])
    car_resolution = config.get("car_resolution_arcmin", None)
    if car_resolution is not None:
        car_resolution = car_resolution * u.arcmin
    nside, shape, wcs = get_map_shape(
        ch=channels[0],
        nside=config.get("nside", None),
        car_resolution=car_resolution,
        car=car,
    )

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
                if function_accepts_argument(comp_class, "shape") and shape is not None:
                    comp_config["shape"] = shape
                    comp_config["wcs"] = wcs
                components[component_type][comp_name] = comp_class(
                    nside=nside, **comp_config
                )

    map_sim = MapSim(
        channels=config["channels"],
        nside=nside,
        car=car,
        car_resolution=car_resolution,
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
        instrument_parameters=config["instrument_parameters"],
    )
    return map_sim


class MapSim:
    def __init__(
        self,
        channels,
        nside=None,
        car=False,
        car_resolution=None,
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
        instrument_parameters=DEFAULT_INSTRUMENT_PARAMETERS,
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
            If "all", all channels are included.
            Otherwise a list of channel tags:
            LT1_UHF1 and LT0_UHF1 = "LT1_UHF1,LT0_UHF1"
            Otherwise, provide a string with:
            * a key, e.g. tube or telescope
            * :
            * comma separated list of desider values
            e.g. all SAT channels = "telescope:SA"
            LT1 and LT2 tubes = "tube:LT1,LT2"
        nside : int
            output HEALPix Nside, if None, automatically pick the default resolution of the
            first channel,
            see https://github.com/simonsobs/mapsims/tree/master/mapsims/data/so_default_resolution.csv
        car : bool
            True for CAR, False for HEALPix
        car_resolution : astropy.Quantity
            CAR pixels resolution with angle unit
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
        instrument_parameters : ascii file in IPAC table format path or str
            A string specifies an instrument parameters file
            included in the package `data/` folder
            A path or a string containing a path to an externally provided IPAC table file with
            the expected format. By default the latest Simons Observatory parameters
            Instrument parameters in IPAC ascii format, one channel per row, with columns (with units):
                tag, band, center_frequency, fwhm
            It also assumes that in the same folder there are IPAC table files named bandpass_{tag}.tbl
            with columns:
                bandpass_frequency, bandpass_weight


        """

        self.channels = parse_channels(
            instrument_parameters=instrument_parameters, filter=channels
        )

        self.car = car
        self.nside, self.shape, self.wcs = get_map_shape(
            ch=self.channels[0],
            nside=nside,
            car_resolution=car_resolution,
            car=self.car,
        )

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
            output_map_shape = (len(ch), 3) + self.shape
            output_map = np.zeros(output_map_shape, dtype=np.float64)
            if self.run_pysm:
                for each, channel_map in zip(ch, output_map):
                    bandpass_integrated_map = self.pysm_sky.get_emission(
                        *each.bandpass
                    ).value
                    beam_width_arcmin = each.beam
                    # smoothing and coordinate rotation with 1 spherical harmonics transform
                    smoothed_map = hp.ma(
                        pysm.apply_smoothing_and_coord_transform(
                            bandpass_integrated_map,
                            fwhm=beam_width_arcmin,
                            lmax=3 * self.nside - 1,
                            rot=None
                            if input_reference_frame == self.pysm_output_reference_frame
                            else hp.Rotator(
                                coord=(
                                    input_reference_frame,
                                    self.pysm_output_reference_frame,
                                )
                            ),
                            map_dist=None
                            if COMM_WORLD is None
                            else pysm.MapDistribution(
                                nside=self.nside,
                                smoothing_lmax=3 * self.nside - 1,
                                mpi_comm=COMM_WORLD,
                            ),
                        )
                    )

                    if smoothed_map.shape[0] == 1:
                        channel_map[0] += smoothed_map
                    else:
                        channel_map += smoothed_map

            output_map = output_map.reshape((len(ch), 1, 3) + self.shape)

            if self.other_components is not None:
                for comp in self.other_components.values():
                    kwargs = dict(tube=ch[0].tube, output_units=self.unit)
                    if function_accepts_argument(comp.simulate, "ch"):
                        kwargs.pop("tube")
                        kwargs["ch"] = ch
                    if function_accepts_argument(comp.simulate, "nsplits"):
                        kwargs["nsplits"] = self.nsplits
                    if function_accepts_argument(comp.simulate, "seed"):
                        kwargs["seed"] = self.num
                    component_map = comp.simulate(**kwargs)
                    if self.nsplits == 1:
                        component_map = component_map.reshape(
                            (len(ch), 1, 3) + self.shape
                        )
                    component_map[hp.mask_bad(component_map)] = np.nan
                    output_map = output_map + component_map

            for each, channel_map in zip(ch, output_map):
                if write_outputs:
                    for split, each_split_channel_map in enumerate(channel_map):
                        filename = self.output_filename_template.format(
                            telescope=each.telescope
                            if each.tube is None
                            else each.tube,
                            band=each.band,
                            nside=self.nside,
                            tag=self.tag,
                            num=self.num,
                            nsplits=self.nsplits,
                            split=split + 1,
                        )
                        warnings.warn("Writing output map " + filename)
                        if self.car:
                            pixell.enmap.write_map(
                                os.path.join(self.output_folder, filename),
                                each_split_channel_map,
                                extra=dict(units=self.unit),
                            )
                        else:
                            each_split_channel_map[
                                np.isnan(each_split_channel_map)
                            ] = hp.UNSEEN
                            each_split_channel_map = hp.reorder(
                                each_split_channel_map, r2n=True
                            )
                            hp.write_map(
                                os.path.join(self.output_folder, filename),
                                each_split_channel_map,
                                coord=self.pysm_output_reference_frame,
                                column_units=self.unit,
                                dtype=np.float32,
                                overwrite=True,
                                nest=True,
                            )
                else:
                    if self.nsplits == 1:
                        channel_map = channel_map[0]
                    if not self.car:
                        channel_map[np.isnan(channel_map)] = hp.UNSEEN
                    output[each.tag] = channel_map
        if not write_outputs:
            return output
