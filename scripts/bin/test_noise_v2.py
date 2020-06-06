from __future__ import print_function
from orphics import maps, io, cosmology, stats
from pixell import enmap, bunch, curvedsky as cs
import numpy as np
import os, sys
from mapsims import noise  # noise_car_fix as noise
import mapsims
import healpy as hp
from mapsims import so_utils
from enlib import bench

"""
Integration tests for mapsims.noise with v0.2 hitmaps.
"""


def config_from_yaml(filename):
    import yaml
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config


def get_geom(c):
    try:
        nside = c.nside
        shape, wcs = None, None
        res = None
    except:
        nside = None
        res = c.res
        shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res / 60.0))
    return nside, shape, wcs, res


config = config_from_yaml("bin/tests.yml")


def get_spectra(tube, omap, nside, res, white,w2):
    zeros = []
    if nside is not None:
        mlmax = nside * 2
        alms = [hp.map2alm(omap[i, 0, ...], lmax=mlmax, pol=True)/np.sqrt(w2) for i in range(2)]
    else:
        mlmax = 4096 * 2 / res
        alms = [cs.map2alm(omap[i, 0, ...], lmax=mlmax, spin=[0, 2])/np.sqrt(w2) for i in range(2)]

    for i in range(2):
        for j in range(i, 2):
            for p in range(3):
                for q in range(p + 1, 3):
                    zeros.append(hp.alm2cl(alms[i][p], alms[j][q]))

    for i in range(1, 3):
        zeros.append(hp.alm2cl(alms[0][i], alms[1][i]))

    c11 = [hp.alm2cl(alms[0][i], alms[0][i]) for i in range(3)]
    c22 = [hp.alm2cl(alms[1][i], alms[1][i]) for i in range(3)]
    cross = hp.alm2cl(alms[0][0], alms[1][0])
    if white or tube[0] == "S":
        zeros.append(cross)
        c12 = []
    else:
        c12 = [cross]
    ls = np.arange(c11[0].size)
    return ls, c11, c22, c12, zeros


for key in config.keys():
    print(f"Test {key}")
    c = bunch.Bunch(config[key])
    nside, shape, wcs, res = get_geom(c)
    nsim = noise.SONoiseSimulator(
        nside=nside,
        shape=shape,
        wcs=wcs,
        homogenous=c.homogenous,
        sky_fraction=c.fsky if c.homogenous else None,
    )
    ells, nlt, nlp = nsim.get_noise_spectra(c.tube, ncurve_sky_fraction=c.fsky)

    if not(c.homogenous):
        try:
            if nside is not None:
                mask = hp.read_map(c.mask)
            else:
                mask = enmap.read_map(c.mask)
        except:
            mask = None
    else:
        mask = None
        

    with bench.show("sim"):
        omap = nsim.simulate(c.tube, seed=(1,), nsplits=1, atmosphere=not(c.wnoise), hitmap=mask)

    if mask is None:
        mask = 1
        w2 = 1
    else:
        from solenspipe import wfactor
        w2 = wfactor(2,mask,equal_area=not(nside is None))

    ls, c11, c22, c12, zeros = get_spectra(c.tube, omap*mask, nside, res, c.wnoise,w2)
    pl = io.Plotter("Cell", xyscale="loglin")
    [pl.add(ls, zero) for zero in zeros]
    pl.done(f"{key}_zeros.png")

    pl = io.Plotter("Cell", xyscale="linlog")
    pl.add(ls, c11[0], color="red", alpha=0.5)
    pl.add(ells, nlt[0], color="red", ls="--")
    pl.add(ls, c22[0], color="blue", alpha=0.5)
    pl.add(ells, nlt[1], color="blue", ls="--")
    for c in c12:
        pl.add(ls, c, color="green", alpha=0.5)
        pl.add(ells, nlt[2], color="green", ls="--")
    pl._ax.set_ylim(1e-5, 1e3)
    pl.done(f"{key}_cs_TT.png")

    pl = io.Plotter("Cell", xyscale="linlog")
    pl.add(ls, c11[1], color="red", alpha=0.5)
    pl.add(ells, nlp[0], color="red", ls="--")
    pl.add(ls, c22[1], color="blue", alpha=0.5)
    pl.add(ells, nlp[1], color="blue", ls="--")
    pl._ax.set_ylim(1e-5, 1e3)
    pl.done(f"{key}_cs_EE.png")

    pl = io.Plotter("Cell", xyscale="linlog")
    pl.add(ls, c11[2], color="red", alpha=0.5)
    pl.add(ells, nlp[0], color="red", ls="--")
    pl.add(ls, c22[2], color="blue", alpha=0.5)
    pl.add(ells, nlp[1], color="blue", ls="--")
    pl._ax.set_ylim(1e-5, 1e3)
    pl.done(f"{key}_cs_BB.png")
