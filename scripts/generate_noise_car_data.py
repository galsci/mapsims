import numpy as np
import healpy as hp

import pytest
from astropy.tests.helper import assert_quantity_allclose

from astropy.utils import data
import mapsims
import pysm.units as u
from mapsims import so_utils
from pixell import enmap
from orphics import io

res = np.deg2rad(30 / 60.) 

seed = 1234
shape,wcs = enmap.fullsky_geometry(res=res)
simulator = mapsims.SONoiseSimulator(shape=shape,wcs=wcs)
for tube in ["LT0", "ST3"]:
    output_map = simulator.simulate(tube,seed=seed)
    fname = f"noise_{tube}_uKCMB_classical_res30_seed1234_car"
    expected_map = enmap.write_map(
        f"{fname}.fits",
        output_map
    )


nside = 16
simulator = mapsims.SONoiseSimulator(nside=nside)
for tube in ["ST0", "ST3"]:
    output_map = simulator.simulate(tube,seed=seed)
    for i,band in enumerate(so_utils.tubes[tube]):
        fname = f"noise_{tube}_{band}_uKCMB_classical_nside16_seed1234_healpix"
        expected_map = hp.write_map(
            f"{fname}.fits",
            output_map[i,0]
        )
