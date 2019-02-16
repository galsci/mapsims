import os.path
import importlib
import numpy as np

import pysm
import configobj

import healpy as hp

from . import so_utils
from . import Channel

PYSM_COMPONENTS = {
    comp[0]: comp for comp in ["synchrotron", "dust", "freefree", "cmb", "ame"]
}
default_output_filename_template="simonsobs_{telescope}{band:03d}_nside{nside}.fits"

def command_line_script(args=None):

    import argparse

    parser = argparse.ArgumentParser(
        description="Execute map based simulations for Simons Observatory"
    )
    parser.add_argument("config", type=str, help="Configuration file", nargs='+',)
    res = parser.parse_args(args)
    simulator = from_config(res.config)
    simulator.execute(write_outputs=True)


def import_class_from_string(class_string):
    module_name, class_name = class_string.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def from_config(config_file):
    if isinstance(config_file, str):
        config_file = [config_file]

    config = configobj.ConfigObj(config_file[0], interpolation=False)

    for other_config in config_file[1:]:
        config.merge(configobj.ConfigObj(other_config, interpolation=False))

    config = configobj.ConfigObj(config, interpolation="Template")

    pysm_components_string = None

    components = {}
    for component_type in ["pysm_components", "other_components"]:
        components[component_type] = {}
        if component_type in config.sections:
            component_type_config = config[component_type]
            if component_type == "pysm_components":
                pysm_components_string = component_type_config.pop(
                    "pysm_components_string", default=None
                )
            for comp_name in component_type_config:
                comp_config = component_type_config[comp_name]
                comp_class = import_class_from_string(comp_config.pop("class"))
                for k, v in comp_config.items():
                    try:
                        if "." in v:
                            comp_config[k] = float(v)
                        else:
                            comp_config[k] = int(v)
                    except ValueError:
                        if v == "True":
                            comp_config[k] = True
                        elif v == "False":
                            comp_config[k] = False
                components[component_type][comp_name] = comp_class(
                    **(comp_config.dict())
                )

    map_sim = MapSim(
        channels=config["channels"],
        nside=int(config["output_nside"]),
        unit=config["unit"],
        output_folder=config.get("output_folder", "output"),
        output_filename_template=config.get("output_filename_template", default_output_filename_template),
        pysm_components_string=pysm_components_string,
        pysm_custom_components=components["pysm_components"],
        other_components=components["other_components"],
    )
    return map_sim


class MapSim:
    def __init__(
        self,
        channels,
        nside,
        unit="uK_CMB",
        output_folder="output",
        output_filename_template=default_output_filename_template,
        pysm_components_string=None,
        pysm_custom_components=None,
        other_components=None,
    ):

        if channels in ["LA", "SA"]:
            self.channels = [
                Channel(channels, band) for band in so_utils.get_bands(channels)
            ]
        elif channels in ["all", "SO"]:
            self.channels = [
                Channel(telescope, band)
                for telescope in ["LA", "SA"]
                for band in so_utils.get_bands(telescope)
            ]
        else:
            self.channels = []
            if isinstance(channels, str):
                channels = [channels]
            for ch in channels:
                [telescope, str_band] = ch.split("_")
                self.channels.append(Channel(telescope, int(str_band)))

        self.bands = np.unique([ch.band for ch in self.channels])
        self.nside = nside
        self.unit = unit
        self.pysm_components_string = pysm_components_string
        self.pysm_custom_components = pysm_custom_components
        self.run_pysm = not (
            (pysm_components_string is None)
            and (pysm_custom_components is None or len(pysm_custom_components) == 0)
        )
        self.other_components = other_components
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.output_folder = output_folder
        self.output_filename_template = output_filename_template

    def execute(self, write_outputs=False):

        if self.run_pysm:
            sky_config = {}
            if self.pysm_components_string is not None:
                for model in self.pysm_components_string.split(","):
                    sky_config[PYSM_COMPONENTS[model[0]]] = pysm.nominal.models(
                        model, self.nside
                    )

            self.pysm_sky = pysm.Sky(sky_config)

            if self.pysm_custom_components is not None:
                for comp_name, comp in self.pysm_custom_components.items():
                    self.pysm_sky.add_component(comp_name, comp)

        if not write_outputs:
            output = {}

        for band in self.bands:

            instrument = {
                "frequencies": np.array([band]),
                "nside": self.nside,
                "use_bandpass": False,
                "add_noise": False,
                "output_units": self.unit,
                "use_smoothing": False,
            }

            if self.run_pysm:
                instrument = pysm.Instrument(instrument)
                band_map = hp.ma(
                    instrument.observe(self.pysm_sky, write_outputs=False)[0][0]
                )

                assert band_map.ndim == 2
                assert band_map.shape[0] == 3

            for ch in self.channels:
                if ch.band == band:
                    if self.run_pysm:
                        beam_width_arcmin = so_utils.get_beam(ch.telescope, ch.band)
                        output_map = hp.smoothing(
                            band_map, fwhm=np.radians(beam_width_arcmin / 60)
                        )
                    else:
                        output_map = np.zeros(
                            (3, hp.nside2npix(self.nside)), dtype=np.float64
                        )

                    for comp in self.other_components.values():
                        output_map += hp.ma(comp.simulate(ch, output_units=self.unit))

                    if write_outputs:
                        hp.write_map(
                            os.path.join(
                                self.output_folder,
                                self.output_filename_template.format(
                                    telescope=ch.telescope.lower(),
                                    band=ch.band,
                                    nside=self.nside,
                                ),
                            ),
                            output_map,
                            overwrite=True,
                        )
                    else:
                        output[ch] = output_map.filled()
        if not write_outputs:
            return output
