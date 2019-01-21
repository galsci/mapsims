import sys
sys.path.append("../..")

import numpy as np
import SO_Noise_Calculator_Public_20180822 as so_n
import healpy as hp
import matplotlib.pyplot as plt

nside = 16
lmax = 500
seed = 1234
channel = 2

nh_LA = hp.ud_grade(
    hp.read_map("../../data/total_hits_LA_classical.fits.gz", verbose=False),
    nside_out=nside,
)
nh_LA /= np.amax(nh_LA)
fsky_LA = np.mean(nh_LA)
ls_LA, Nl_LA_T, Nl_LA_P, _ = so_n.Simons_Observatory_V3_LA_noise(1, fsky_LA, lmax, 1)
# Extending to l=0
Nl_LA_Tb = np.zeros([len(Nl_LA_T), len(ls_LA) + 2])
Nl_LA_Tb[:, 2:] = Nl_LA_T
Nl_LA_Tb[:, :2] = 0.
Nl_LA_T = Nl_LA_Tb
Nl_LA_Pb = np.zeros([len(Nl_LA_P), len(ls_LA) + 2])
Nl_LA_Pb[:, 2:] = Nl_LA_P
Nl_LA_Pb[:, :2] = 0.
Nl_LA_P = Nl_LA_Pb
ls_LA = np.arange(len(ls_LA) + 2)

nh_SA = hp.ud_grade(
    hp.read_map("../../data/total_hits_SA_classical.fits.gz", verbose=False),
    nside_out=nside,
)
nh_SA /= np.amax(nh_SA)
fsky_SA = np.mean(nh_SA)
ls_SA, Nl_SA_P, _ = so_n.Simons_Observatory_V3_SA_noise(1, 1, 1., fsky_SA, lmax, 1)
# Extending to l=0
Nl_SA_Pb = np.zeros([len(Nl_SA_P), len(ls_SA) + 2])
Nl_SA_Pb[:, 2:] = Nl_SA_P
Nl_SA_Pb[:, :2] = 0.
Nl_SA_P = Nl_SA_Pb
Nl_SA_T = Nl_SA_P / 2.
ls_SA = np.arange(len(ls_SA) + 2)

# Create maps
np.random.seed(seed)
zeros = np.zeros(len(ls_LA))
t_LA, q_LA, u_LA = hp.synfast(
    [Nl_LA_T[channel], Nl_LA_P[channel], Nl_LA_P[channel], zeros, zeros, zeros],
    nside=nside,
    pol=True,
    new=True,
    verbose=False,
)
goodpix = np.where(nh_LA > 0)
badpix = np.where(nh_LA <= 0)
for m in [t_LA, q_LA, u_LA]:
    m[badpix] = 0
    m[goodpix] /= np.sqrt(nh_LA[goodpix])

np.random.seed(seed)
t_SA, q_SA, u_SA = hp.synfast(
    [Nl_SA_T[channel], Nl_SA_P[channel], Nl_SA_P[channel], zeros, zeros, zeros],
    nside=nside,
    pol=True,
    new=True,
    verbose=False,
)
goodpix = np.where(nh_SA > 0)
badpix = np.where(nh_SA <= 0)
for m in [t_SA, q_SA, u_SA]:
    m[badpix] = 0
    m[goodpix] /= np.sqrt(nh_SA[goodpix])

# Write maps
hp.write_map(
    "noise_SA_uKCMB_classical_nside%d_channel%d_seed%d.fits" % (nside, channel, seed),
    [t_SA, q_SA, u_SA],
)
hp.write_map(
    "noise_LA_uKCMB_classical_nside%d_channel%d_seed%d.fits" % (nside, channel, seed),
    [t_LA, q_LA, u_LA],
)
