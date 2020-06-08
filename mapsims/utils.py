# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions.

import os
from astropy.utils import data
import warnings

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
                warnings.warn(f"Access data from {full_path}")
                return full_path
        with data.conf.set_temp("dataurl", self.data_url), data.conf.set_temp(
            "remote_timeout", 30
        ):
            warnings.warn(f"Retrieve data for {filename} (if not cached already)")
            map_out = data.get_pkg_data_filename(filename, show_progress=True)
        return map_out
