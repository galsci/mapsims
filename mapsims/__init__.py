# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .channel_utils import Channel, parse_channels
from .noise import SONoiseSimulator
from .alms import PrecomputedAlms
from .cmb import PrecomputedCMB, StandalonePrecomputedCMB
from .runner import MapSim, from_config, get_default_so_resolution

# Enforce Python version check during package import.
import sys

__minimum_python_version__ = "3.7"


class UnsupportedPythonError(Exception):
    pass


if sys.version_info < tuple(
    (int(val) for val in __minimum_python_version__.split("."))
):
    raise UnsupportedPythonError(
        "mapsims does not support Python < {}".format(__minimum_python_version__)
    )

__version__ = "2.5.0"
