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
from mpi4py import MPI


def mpi_distribute(num_tasks,avail_cores,allow_empty=False):
    if not(allow_empty): assert avail_cores<=num_tasks
    min_each, rem = divmod(num_tasks,avail_cores)
    num_each = np.array([min_each]*avail_cores) # first distribute equally
    if rem>0: num_each[-rem:] += 1  # add the remainder to the last set of cores (so that rank 0 never gets extra jobs)
    task_range = list(range(num_tasks)) # the full range of tasks
    cumul = np.cumsum(num_each).tolist() # the end indices for each task
    task_dist = [task_range[x:y] for x,y in zip([0]+cumul[:-1],cumul)] # a list containing the tasks for each core
    assert sum(num_each)==num_tasks
    assert len(num_each)==avail_cores
    assert len(task_dist)==avail_cores
    return num_each,task_dist

def distribute(njobs,verbose=True,**kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numcores = comm.Get_size()
    num_each,each_tasks = mpi_distribute(njobs,numcores,**kwargs)
    if rank==0: print ("At most ", max(num_each) , " tasks...")
    my_tasks = each_tasks[rank]
    return comm,rank,my_tasks


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


# This needs to be obtained from pwg-scripts or something similar
resolutions = {}
bounds = {}
resolutions["LA"] = [2.0, 0.5]
resolutions["SA"] = [12.0, 4.0]
bounds["LA"] = [-80.0, 40.0]
bounds["SA"] = [-70.0, 30.0]



hp_remote_data = mutils.RemoteData(healpix=True, version=version)
cr_remote_data = mutils.RemoteData(healpix=False, version=version)
assert (
    cr_remote_data.local_folder is not None
), f"Output directory {cr_remote_data.local_folder} needs to exist."


def save_maps(hmap,root_name,telescope):
    for res in resolutions[telescope]:
        shape, wcs = get_geometry(res, bounds_deg=bounds[telescope])
        imap = ivar_hp_to_cyl(hmap, shape, wcs)
        oname = cr_remote_data.get_local_output(
            f"{root_name}_CAR_{res:.2f}_arcmin.fits"
        )
        enmap.write_map(oname, imap)
        os.system(f"gzip -f {oname}")
        plots = enplot.get_plots(imap)
        savename = cr_remote_data.get_local_output(f"rmap_{root_name}_{res:.2f}")
        enplot.write(
            savename,
            plots
        )
        print(f"Plot saved to {savename}")


if version=='v0.1':
    telescopes = ["SA", "LA"]
    modes = ["opportunistic", "classical"]

    jobs = []
    for telescope in telescopes:
        for mode in modes:
            jobs.append((telescope,mode))

    comm,rank,my_tasks = distribute(len(jobs))

    for task in my_tasks:
        telescope,mode = jobs[task]
        root_name = f'total_hits_{telescope}_{mode}'
        hmap = hp.read_map(hp_remote_data.get(f"{root_name}.fits.gz"))
        save_maps(hmap,root_name,telescope)

elif version=='v0.2':
    from mapsims import so_utils
    tubes = so_utils.tubes
    modes = []
    for tube in tubes.keys():
        for band in tubes[tube]: modes.append( f'{tube}_{band}' ) 

    comm,rank,my_tasks = distribute(len(modes))

    for task in my_tasks:
        mode = modes[task]
        telescope = f'{mode[0]}A'
        root_name = f'{mode}_01_of_20.nominal_telescope_all_time_all_hmap'
        hmap = hp.read_map(hp_remote_data.get(f"{root_name}.fits.gz"))
        save_maps(hmap,root_name,telescope)
    
