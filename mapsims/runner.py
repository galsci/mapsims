import importlib
import numpy as np

import pysm
import configobj

import healpy as hp

from . import so_utils

PYSM_COMPONENTS = {
    comp[0]: comp for comp in ["synchrotron", "dust", "freefree", "cmb", "ame"]
}


def import_class_from_string(class_string):
    module_name, class_name = class_string.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def from_config(config_file):
    config = configobj.ConfigObj(config_file, interpolation="Template")

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
                        pass
                components[component_type][comp_name] = comp_class(**comp_config)

    map_sim = MapSim(
        config["SO_telescope"],
        int(config["SO_band"]),
        int(config["output_nside"]),
        config["unit"],
        pysm_components_string=pysm_components_string,
        pysm_custom_components=components["pysm_components"],
        other_components=components["other_components"],
    )
    return map_sim


class MapSim:

    def __init__(
        self,
        telescope,
        band,
        nside,
        unit="uK_CMB",
        pysm_components_string=None,
        pysm_custom_components=None,
        other_components=None,
    ):
        self.telescope = telescope
        self.band = band
        self.beam_width_arcmin = so_utils.get_beam(self.telescope, self.band)
        self.nside = nside
        self.unit = unit
        self.pysm_components_string = pysm_components_string
        self.pysm_custom_components = pysm_custom_components
        self.other_components = other_components

    def execute(self, seed=None):

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

        instrument = {
            "frequencies": np.array([self.band]),
            "nside": self.nside,
            "use_bandpass": False,
            "add_noise": False,
            "output_units": self.unit,
            "use_smoothing": False,
        }

        instrument = pysm.Instrument(instrument)
        output_map = hp.ma(instrument.observe(self.pysm_sky, write_outputs=False)[0][0])

        assert output_map.ndim == 2
        assert output_map.shape[0] == 3

        output_map = hp.smoothing(
            output_map, fwhm=np.radians(self.beam_width_arcmin / 60)
        )

        for comp in self.other_components.values():
            output_map += hp.ma(comp.simulate())
        return output_map.filled()
