# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:35:29 2018

@author: Stavros
"""

import numpy as np
from plot_directories import multiple_exponents_dir
from scipy.stats import linregress

### !!! .NPY DESCRIPTION !!! ###
# obs = (Calculations, 12, Upsamplings+1)
# ind1: different calculations
# ind2: [Mag, En, Susc, specHeat, Mag2, Mag4, En2, tpf(L/2), tpf(L/4), S0, S1, S2]
# ind3: different lengths

NAME = 'Simple2D16relu_L2_64_32_K333_PBC_C1UP3VER1'
NAME = 'Simple2D16relu_L2_64_16_16_K3333_PBC_C42UP3_old'

obs = np.load('%s/%s.npy'%(multiple_exponents_dir, NAME))

calcs, n_obs, upsamplings = obs.shape
upsamplings += -1
L_list = 2**np.arange(4, upsamplings+5)

beta, gamma = np.zeros(calcs), np.zeros(calcs)
eta1, eta2 = np.zeros(calcs), np.zeros(calcs)
for iC in range(calcs):
    beta[iC] = linregress(np.log10(L_list), np.log10(obs[iC, 0]))[0]
    gamma[iC] = linregress(np.log10(L_list), np.log10(obs[iC, 2]))[0]
    eta1[iC] = linregress(np.log10(L_list/2.0), np.log10(obs[iC, 7]))[0]
    eta2[iC] = linregress(np.log10(L_list/4.0), np.log10(obs[iC, 8]))[0]

print('Calculations: %d  -  Upsamplings: %d'%(calcs, upsamplings))
print('Beta: %.6f  +-  %.6f'%(beta.mean(), beta.std()))
print('Gamma: %.6f  +-  %.6f'%(gamma.mean(), gamma.std()))
print('Eta1: %.6f  +-  %.6f'%(eta1.mean(), eta1.std()))
print('Eta2: %.6f  +-  %.6f'%(eta2.mean(), eta2.std()))