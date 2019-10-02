from __future__ import print_function
import matplotlib
from orphics import io # msyriac/orphics
import numpy as np
import os,sys
from soapack import interfaces as sints # simonsobs/soapack
from tilec import utils as tutils # ACTCollaboration/tilec

defsamp = 4
fnames = []
qids = "boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08,p09".split(',')
nsamps = [None,None,defsamp,defsamp,None,None,defsamp,None,None,defsamp,defsamp,defsamp,defsamp]

flist = [30,44,70,100,143,217,353,545,857]

ftsize = 18
pl = io.Plotter(xyscale='loglin',xlabel='$\\nu$ (GHz)',ylabel='$T(\\nu)$',figsize=(10,3),ftsize=ftsize)

def add_plot(nu,bp,col,label,nsamp = defsamp):
    global flist
    bnp = bp/bp.max()
    bnp[bnp<5e-4] = np.nan
    pl.add(nu,bnp,color=col,lw=2,label=label)

    if nsamp is not None:
        fsel = nu[bnp>4e-1]
        fmin = fsel.min()
        fmax = fsel.max()
        #flist = flist + np.geomspace(fmin,fmax,nsamp).tolist()
        flist = flist + np.linspace(fmin,fmax,nsamp).tolist()

    


nu,bp30,bp44 = np.loadtxt("../bandpasses/LF/LF_skinnyfatter.txt",unpack=True,skiprows=1)
add_plot(nu,bp30,"red",None)
add_plot(nu,bp44,"red",None)
nu,bp30,bp44 = np.loadtxt("../bandpasses/MF/MF_wOMT_wLPF_v1_LAT_beam.txt",unpack=True,skiprows=1)
add_plot(nu,bp30,"red",None,None)
add_plot(nu,bp44,"red",None,None)
nu,bp30,bp44 = np.loadtxt("../bandpasses/MF/MF_wOMT_wLPF_v1_SAT_beam.txt",unpack=True,skiprows=1)
add_plot(nu,bp30,"red",None,None)
add_plot(nu,bp44,"red",None,None)
nu,bp30,bp44 = np.loadtxt("../bandpasses/UHF/UHF_v3.7_wOMT_scaled_2019.txt",unpack=True,skiprows=1)
add_plot(nu,bp30,"red",None,None)
add_plot(nu,bp44,"red",None)
    

lfi_done = False
hfi_done = False
act_done = False
for i,qid in enumerate(qids):
    dm = sints.models[sints.arrays(qid,'data_model')]()

    if dm.name=='act_mr3':
        season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
        array = '_'.join([array1,array2])
    elif dm.name=='planck_hybrid':
        season,patch,array = None,None,sints.arrays(qid,'freq')

    fname = "../tile-c/data/"+dm.get_bandpass_file_name(array)
    if fname in fnames: continue
    fnames.append(fname)
    print(fname)
    nu,bp = np.loadtxt(fname,unpack=True,usecols=[0,1])
    
    if tutils.is_lfi(qid):
        col = 'C0'
        if not(lfi_done): label = 'LFI'
        else: label = None
        lfi_done = True
    elif tutils.is_hfi(qid):
        col = 'C1'
        if not(hfi_done): label = 'HFI'
        else: label = None
        hfi_done = True
    else:
        col = 'C2'
        if not(act_done): label = 'ACT'
        else: label = None
        act_done = True

    label = None

    add_plot(nu,bp,col,label,nsamps[i])


# New list after trimming manually. Uncomment this after copying the result of the script here.
flist = [21.6, 24.47, 27.33, 30, 35.93, 41.67, 44, 47.4, 63.9, 67.8, 70, 73.7, 79.6, 90.16, 100, 111.49, 128.99, 143, 152.65, 164.48, 188.56, 210.29, 217, 232.03, 256.0, 275.17, 294.33, 305.84, 313.5, 340.21, 353, 374.59, 408.96, 466.56, 525.36, 545, 584.17, 642.97, 728.93, 817.39, 857, 905.86, 994.32]
    
for f in flist:
    pl.vline(f)

print(len(flist),[round(x,2) for x in sorted(flist)])
    
    
pl._ax.set_xlim(20,1200)
pl.hline()
pl._ax.set_xticks([30,44,70,100,150, 217, 353,545])
pl._ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
pl.done("fig_bandpass.png")
