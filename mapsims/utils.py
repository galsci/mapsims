# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions.

import os
from astropy.utils import data
import logging
log = logging.getLogger("mapsims")

try:
    from collections import Mapping
except ImportError: # Python 3.10
    from collections.abc import Mapping

DEFAULT_INSTRUMENT_PARAMETERS = "simonsobs_instrument_parameters_2020.06"


def _DATAURL(healpix, version):
    if healpix:
        return (
            f"https://portal.nersc.gov/project/sobs/so_mapsims_data/{version}/healpix/"
        )
    else:
        return f"https://portal.nersc.gov/project/sobs/so_mapsims_data/{version}/car/"


def _PREDEFINED_DATA_FOLDERS(healpix, version):
    if healpix:
        return [
            f"/global/cfs/cdirs/sobs/www/so_mapsims_data/{version}/healpix",  # NERSC
            f"/scratch/r/rbond/msyriac/data/sobs/so_mapsims_data/{version}/healpix",  # SCINET/NIAGARA
        ]
    else:
        return [
            f"/global/cfs/cdirs/sobs/www/so_mapsims_data/{version}/car",  # NERSC
            f"/scratch/r/rbond/msyriac/data/sobs/so_mapsims_data/{version}/car",  # SCINET/NIAGARA
        ]


class RemoteData:
    def __init__(self, version, healpix):
        """Access noise template from remote server

        Noise templates are stored on the CMB project space at NERSC
        and are made available via web.
        The get method of this class tries to access data locally from one
        of the PREDEFINED_DATA_FOLDERS defined above, if it fails, it
        retrieves the files and caches them remotely using facilities
        provided by `astropy.utils.data`.

        This class is copied and modified from so_pysm_models.

        Parameters
        ----------

        healpix : boolean
            Whether the noise templates are in healpix or CAR.
        version : string
            Version identifier.
        """
        self.data_url = _DATAURL(healpix, version)
        self.data_folders = _PREDEFINED_DATA_FOLDERS(healpix, version)
        self.local_folder = None
        for folder in self.data_folders:
            if os.path.exists(folder):
                self.local_folder = folder

    def get_local_output(self, filename):
        assert self.local_folder is not None
        return os.path.join(self.local_folder, filename)

    def get(self, filename):
        for folder in self.data_folders:
            full_path = os.path.join(folder, filename)
            if os.path.exists(full_path):
                log.warn(f"Access data from {full_path}")
                return full_path
        with data.conf.set_temp("dataurl", self.data_url), data.conf.set_temp(
            "remote_timeout", 90
        ):
            log.warn(f"Retrieve data for {filename} (if not cached already)")
            map_out = data.get_pkg_data_filename(filename, show_progress=True)
        return map_out


import collections


def merge_dict(d1, d2):
    """
    Modifies d1 in-place to contain values from d2.  If any value
    in d1 is a dictionary (or dict-like), *and* the corresponding
    value in d2 is also a dictionary, then merge them in-place.
    """
    for k, v2 in d2.items():
        v1 = d1.get(k)  # returns None if v1 has no value for this key
        if isinstance(v1, Mapping) and isinstance(v2, Mapping):
            merge_dict(v1, v2)
        else:
            d1[k] = v2
