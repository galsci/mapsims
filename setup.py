#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst

import builtins

# Ensure that astropy-helpers is available
import ah_bootstrap  # noqa

from setuptools import setup
from setuptools.config import read_configuration

from astropy_helpers.setup_helpers import register_commands, get_package_info
from astropy_helpers.version_helpers import generate_version_py

conf = read_configuration("setup.cfg")
# Store the package name in a built-in variable so it's easy
# to get from other parts of the setup infrastructure
builtins._ASTROPY_PACKAGE_NAME_ = conf["metadata"]["name"]

# Create a dictionary with setup command overrides. Note that this gets
# information about the package (name and version) from the setup.cfg file.
try:
    cmdclass = register_commands(
        conf["metadata"]["name"],
        conf["metadata"]["version"],
        conf["metadata"]["version"],
    )
except TypeError as e:
    raise TypeError("Need updated version of astropy-helpers") from e

# Freeze build information in version.py. Note that this gets information
# about the package (name and version) from the setup.cfg file.
version = generate_version_py(conf["metadata"]["name"], conf["metadata"]["version"])

# Get configuration information from all of the various subpackages.
# See the docstring for setup_helpers.update_package_files for more
# details.
package_info = get_package_info()

setup(version=version, cmdclass=cmdclass, **package_info)
