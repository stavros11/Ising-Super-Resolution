# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:35:29 2018

@author: Stavros
"""

import numpy as np
from scipy.stats import linregress

# If plot_directories module is available:
from plot_directories import multiple_exponents_dir
# otherwise fix directory

### !!! .NPY DESCRIPTION !!! ###
# obs = (Calculations, 12, Upsamplings+1)
# ind1: different calculations
# ind2: [Mag, En, Susc, specHeat, Mag2, Mag4, En2, tpf(L/2), tpf(L/4), S0, S1, S2]
# ind3: different lengths

# Load data (fix .npy directory here!)
NAME = 'Simple2D16relu_L2_64_32_K333_PBC_C20UP3VER123_CONC'
#NAME = 'Simple2D16relu_L2_64_32_K513_PBC_C20UP3VER1'
obs = np.load('%s/%s.npy'%(multiple_exponents_dir, NAME))

calcs, n_obs, upsamplings = obs.shape
upsamplings += -1
L_list = 2**np.arange(4, upsamplings+5)

## (Calcs, [slope, intercept, rvalue, pvalue, stderr])
beta_lr, gamma_lr = np.zeros([calcs, 5]), np.zeros([calcs, 5])
eta1_lr, eta2_lr = np.zeros([calcs, 5]), np.zeros([calcs, 5])

binder = np.zeros([calcs, upsamplings+1])
for iC in range(calcs):
    beta_lr[iC] = linregress(np.log10(L_list), np.log10(obs[iC, 0]))
    gamma_lr[iC] = linregress(np.log10(L_list), np.log10(obs[iC, 2]))
    eta1_lr[iC] = linregress(np.log10(L_list/2.0), np.log10(obs[iC, 7]))
    eta2_lr[iC] = linregress(np.log10(L_list/4.0), np.log10(obs[iC, 8]))
    
    # See Eq. (76) in Sandvik: https://arxiv.org/pdf/1101.3281.pdf
    binder[iC] = 3.0 * (1 - obs[iC, 5] / obs[iC, 4]**2 / 3.0) / 2.0
    
beta = -beta_lr[:, 0]
gamma= gamma_lr[:, 0]
eta1, eta2 = -eta1_lr[:, 0], -eta2_lr[:, 0]

beta_err = np.abs(beta - 0.125) * 100 / 0.125
gamma_err = np.abs(gamma - 1.75) * 100 / 1.75
eta1_err = np.abs(eta1 - 0.25) * 100 / 0.25
eta2_err = np.abs(eta2 - 0.25) * 100 / 0.25

print('Calculations: %d  -  Upsamplings: %d'%(calcs, upsamplings))
print('Beta: %.6f  +-  %.6f  Error: %.2f%%'%(beta.mean(), beta.std(), beta_err.mean()))
print('Gamma: %.6f  +-  %.6f  Error: %.2f%%'%(gamma.mean(), gamma.std(), gamma_err.mean()))
print('Eta1: %.6f  +-  %.6f  Error: %.2f%%'%(eta1.mean(), eta1.std(), eta1_err.mean()))
print('Eta2: %.6f  +-  %.6f  Error: %.2f%%'%(eta2.mean(), eta2.std(), eta2_err.mean()))
