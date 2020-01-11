"""
Reproject hit maps from healpix to CAR.
"""
from __future__ import print_function
from pixell import enmap, mpi, coordinates, utils, enplot
import numpy as np
import os, sys
import healpy as hp
import mapsims
from mapsims import utils as mutils

import argparse

# Parse command line
parser = argparse.ArgumentParser(description="Do a thing.")
parser.add_argument("version", type=str, help="Version name.")
args = parser.parse_args()
print("Command line arguments are %s." % args)
version = args.version


def get_pixsize_rect(shape, wcs):
    """Return the exact pixel size in steradians for the rectangular cylindrical
    projection given by shape, wcs. Returns area[ny], where ny = shape[-2] is the
    number of rows in the image. All pixels on the same row have the same area."""
    ymin = enmap.sky2pix(shape, wcs, [-np.pi / 2, 0])[0]
    ymax = enmap.sky2pix(shape, wcs, [np.pi / 2, 0])[0]
    y = np.arange(shape[-2])
    x = y * 0
    dec1 = enmap.pix2sky(shape, wcs, [np.maximum(ymin, y - 0.5), x])[0]
    dec2 = enmap.pix2sky(shape, wcs, [np.minimum(ymax, y + 0.5), x])[0]
    area = np.abs((np.sin(dec2) - np.sin(dec1)) * wcs.wcs.cdelt[0] * np.pi / 180)
    return area


def ivar_hp_to_cyl(hmap, shape, wcs):
    comm = mpi.COMM_WORLD
    rstep = 100
    dtype = np.float32
    nside = hp.npix2nside(hmap.size)
    dec, ra = enmap.posaxes(shape, wcs)
    pix = np.zeros(shape, np.int32)
    psi = np.zeros(shape, dtype)
    # Get the pixel area. We assume a rectangular pixelization, so this is just
    # a function of y
    ipixsize = 4 * np.pi / (12 * nside ** 2)
    opixsize = get_pixsize_rect(shape, wcs)
    nblock = (shape[-2] + rstep - 1) // rstep
    for bi in range(comm.rank, nblock, comm.size):
        if bi % comm.size != comm.rank:
            continue
        i = bi * rstep
        rdec = dec[i : i + rstep]
        opos = np.zeros((2, len(rdec), len(ra)))
        opos[0] = rdec[:, None]
        opos[1] = ra[None, :]
        ipos = opos[::-1]
        pix[i : i + rstep, :] = hp.ang2pix(nside, np.pi / 2 - ipos[1], ipos[0])
        del ipos, opos
    for i in range(0, shape[-2], rstep):
        pix[i : i + rstep] = utils.allreduce(pix[i : i + rstep], comm)
    omap = enmap.zeros((1,) + shape, wcs, dtype)
    imap = np.array(hmap).astype(dtype)
    imap = imap[None]
    bad = hp.mask_bad(imap)
    bad |= imap <= 0
    imap[bad] = 0
    del bad
    # Read off the nearest neighbor values
    omap[:] = imap[:, pix]
    omap *= opixsize[:, None] / ipixsize
    # We ignore QU mixing during rotation for the noise level, so
    # it makes no sense to maintain distinct levels for them
    mask = omap[1:] > 0
    omap[1:] = np.mean(omap[1:], 0)
    omap[1:] *= mask
    del mask
    return omap[0]


def get_geometry(res_arcmin, bounds_deg):
    return enmap.band_geometry(
        np.deg2rad(bounds_deg), res=np.deg2rad(res_arcmin / 60.0), proj="car"
    )
    return enmap.fullsky_geometry(res=np.deg2rad(res_arcmin / 60.0), proj="car")


# This needs to be obtained from pwg-scripts or something similar
resolutions = {}
bounds = {}
resolutions["LA"] = [2.0, 0.5]
resolutions["SA"] = [12.0, 4.0]
bounds["LA"] = [-80.0, 40.0]
bounds["SA"] = [-70.0, 30.0]


telescopes = ["SA", "LA"]
modes = ["opportunistic", "classical"]

hp_remote_data = mutils.RemoteData(healpix=True, version=version)
cr_remote_data = mutils.RemoteData(healpix=False, version=version)
assert (
    cr_remote_data.local_folder is not None
), f"Output directory {cr_remote_data.local_folder} needs to exist."

for telescope in telescopes:
    for mode in modes:
        hmap = hp.read_map(hp_remote_data.get(f"total_hits_{telescope}_{mode}.fits.gz"))
        for res in resolutions[telescope]:
            shape, wcs = get_geometry(res, bounds_deg=bounds[telescope])
            imap = ivar_hp_to_cyl(hmap, shape, wcs)
            oname = cr_remote_data.get_local_output(
                f"total_hits_{telescope}_{mode}_CAR_{res:.2f}_arcmin.fits"
            )
            enmap.write_map(oname, imap)
            os.system(f"gzip -f {oname}")
            print(f"{telescope}_{mode}_CAR_{res:.2f}_arcmin")
            plots = enplot.get_plots(imap)
            enplot.write(
                savename,
                cr_remote_data.get_local_output(f"rmap_{telescope}_{mode}_{res:.2f}"),
            )
