from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,utils,curvedsky as cs
import numpy as np
import os,sys
import mapsims
from enlib import bench
import healpy as hp

"""
Integration tests to compare healpix and CAR sims
against reported noise curves.
"""

out_path = "/scratch/r/rbond/msyriac/data/depot/mapsims"

def wfactor(n,mask,sht=True,pmap=None,equal_area=False):
    """
    Approximate correction to an n-point function for the loss of power
    due to the application of a mask.

    For an n-point function using SHTs, this is the ratio of 
    area weighted by the nth power of the mask to the full sky area 4 pi.
    This simplifies to mean(mask**n) for equal area pixelizations like
    healpix. For SHTs on CAR, it is sum(mask**n * pixel_area_map) / 4pi.
    When using FFTs, it is the area weighted by the nth power normalized
    to the area of the map. This also simplifies to mean(mask**n)
    for equal area pixels. For CAR, it is sum(mask**n * pixel_area_map) 
    / sum(pixel_area_map).

    If not, it does an expensive calculation of the map of pixel areas. If this has
    been pre-calculated, it can be provided as the pmap argument.
    
    """
    assert mask.ndim==1 or mask.ndim==2
    if pmap is None: 
        if equal_area:
            npix = mask.size
            pmap = 4*np.pi / npix if sht else enmap.area(mask.shape,mask.wcs) / npix
        else:
            pmap = enmap.pixsizemap(mask.shape,mask.wcs)
    return np.sum((mask**n)*pmap) /np.pi / 4. if sht else np.sum((mask**n)*pmap) / np.sum(pmap)


test_nell_standard = False
if test_nell_standard:
    shape,wcs = enmap.fullsky_geometry(res=2.0 * utils.arcmin)
    noise = 10.0
    lknee = 3000
    alpha = -4
    pmap = maps.psizemap(shape,wcs)
    ivar = maps.ivar(shape,wcs,noise,ipsizemap=pmap)
    imap = maps.modulated_noise_map(ivar,lknee=lknee,alpha=alpha,lmax=4000,lmin=50)
    alm = cs.map2alm(imap*np.sqrt(ivar/pmap),lmax=4000)
    cls = hp.alm2cl(alm)
    ls = np.arange(len(cls))
    N_ell = maps.atm_factor(ls,lknee,alpha) + 1.
    N_ell[ls<50] = 0
    pl = io.Plotter('Cell')
    pl.add(ls,cls)
    pl.add(ls,N_ell,ls='--')
    pl.done(f'{out_path}/mapsims_nells_test.png')



def get_sim(healpix,homogeneous,white,scale_to_rms=None):

    if not(healpix):
        mask = enmap.read_map("/scratch/r/rbond/msyriac/data/depot/actlens/car_mask_lmax_3000_apodized_2.0_deg.fits")[0]
        shape,wcs = mask.shape,mask.wcs
        nside = None
        w2 = wfactor(2,mask)
        map2alm = lambda x,lmax: cs.map2alm(x,lmax=lmax)

    else:
        shape = None
        wcs = None
        mask = hp.read_map("/scratch/r/rbond/msyriac/data/depot/solenspipe/lensing_mask_nside_2048_apodized_4.0.fits")
        nside = 2048
        w2 = wfactor(2,mask,equal_area=True)
        map2alm = lambda x,lmax: hp.map2alm(x,lmax=lmax)

    tube = 'LT2'

    with bench.show("init"):
        nsim = mapsims.SONoiseSimulator \
        ( \
            nside=nside,
            shape=shape,
            wcs=wcs,
            ell_max=None,
            return_uK_CMB=True,
            sensitivity_mode="baseline",
            apply_beam_correction=False,
            apply_kludge_correction=True,
            homogeneous=homogeneous,
            no_power_below_ell=None,
            rolloff_ell=50,
            survey_efficiency=0.2,
            full_covariance=True,
            LA_years=5,
            LA_noise_model="SOLatV3point1",
            elevation=50,
            SA_years=5,
            SA_one_over_f_mode="pessimistic",
            sky_fraction=None,
            cache_hitmaps=True,
            boolean_sky_fraction=False,
        )
    
    
    with bench.show("sim"):
        omap = nsim.simulate(
            tube,
            output_units="uK_CMB",
            seed=None,
            nsplits=1,
            mask_value=0,
            atmosphere=not(white),
            hitmap=None,
            white_noise_rms=scale_to_rms,
        )

    # io.hplot(omap[0][0][0],f'{out_path}/mapsims_sim_homogeneous_{homogeneous}_white_{white}_scale_to_rms_{scale_to_rms}',downgrade=4,grid=True,ticks=20)

    with bench.show("nells"):
        ell, ps_T, ps_P, fsky, wnoise_power, hitmaps = nsim.get_noise_properties(tube, nsplits=1, hitmap=None, white_noise_rms=scale_to_rms,atmosphere=not(white))

    with bench.show("ivar"):
        ivar = nsim.get_inverse_variance(tube, output_units="uK_CMB", hitmap=None, white_noise_rms=scale_to_rms)        

    # Calculate raw power spectrum
    imap = omap[0][0][0]
    imap = imap * mask
        
    alm = map2alm(imap,lmax=4000)
    cls = hp.alm2cl(alm) / w2
    ls = np.arange(len(cls))
    pl = io.Plotter('Cell')
    pl.add(ls,cls)
    pl.add(ell,ps_T[0],ls='--')
    pl.done(f'{out_path}/mapsims_nells_homogeneous_{homogeneous}_white_{white}_scale_to_rms_{scale_to_rms}_healpix_{healpix}.png')


    # Calculate whitened map power spectrum
    imap = np.nan_to_num(omap[0][0][0]) * np.sqrt(ivar[0]/nsim.pixarea_map) * mask
    # io.hplot(imap,f'{out_path}/mapsims_wsim_homogeneous_{homogeneous}_white_{white}_scale_to_rms_{scale_to_rms}',downgrade=4,grid=True,ticks=20)
    
    alm = map2alm(imap,lmax=4000)
    cls = hp.alm2cl(alm) / w2
    ls = np.arange(len(cls))
    pl = io.Plotter('Cell')
    pl.add(ls,cls)
    pl.add(ell,ps_T[0]/wnoise_power[0])
    pl.done(f'{out_path}/mapsims_wnells_homogeneous_{homogeneous}_white_{white}_scale_to_rms_{scale_to_rms}_healpix_{healpix}.png')
    

get_sim(healpix=True,homogeneous=False,white=False,scale_to_rms=5.)
get_sim(healpix=True,homogeneous=True,white=False,scale_to_rms=5.)
get_sim(healpix=True,homogeneous=False,white=True,scale_to_rms=5.)
get_sim(healpix=True,homogeneous=True,white=True,scale_to_rms=5.)

get_sim(healpix=False,homogeneous=False,white=False,scale_to_rms=5.)
get_sim(healpix=False,homogeneous=True,white=False,scale_to_rms=5.)
get_sim(healpix=False,homogeneous=False,white=True,scale_to_rms=5.)
get_sim(healpix=False,homogeneous=True,white=True,scale_to_rms=5.)


get_sim(healpix=True,homogeneous=False,white=False,scale_to_rms=None)
get_sim(healpix=True,homogeneous=True,white=False,scale_to_rms=None)
get_sim(healpix=True,homogeneous=False,white=True,scale_to_rms=None)
get_sim(healpix=True,homogeneous=True,white=True,scale_to_rms=None)

get_sim(healpix=False,homogeneous=False,white=False,scale_to_rms=None)
get_sim(healpix=False,homogeneous=True,white=False,scale_to_rms=None)
get_sim(healpix=False,homogeneous=False,white=True,scale_to_rms=None)
get_sim(healpix=False,homogeneous=True,white=True,scale_to_rms=None)




