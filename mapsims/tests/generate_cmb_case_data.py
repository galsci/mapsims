from __future__ import print_function
from pixell import enmap, powspec, curvedsky
import numpy as np
import os, sys
from mapsims import cmb
import healpy as hp


seed = 10
lmax = 150
iteration_num = 0
cmb_set = 0
lensed = False
aberrated = False
nside = 32

theory_filename = "mapsims/data/test_scalCls.dat"
save_name = "mapsims/data/test_map.fits"
output_directory = "mapsims/data"
output_file = cmb._get_cmb_map_string(
    output_directory, iteration_num, cmb_set, lensed, aberrated
)
ps = powspec.read_spectrum(theory_filename)
alms = curvedsky.rand_alm_healpy(ps, lmax=lmax, seed=seed, dtype=np.complex64)
hp.write_alm(output_file, alms, overwrite=True)
imap = cmb.get_cmb_sky(
    iteration_num=0,
    nside=nside,
    cmb_dir=output_directory,
    lensed=False,
    aberrated=False,
)
hp.write_map(save_name, imap, overwrite=True)
