# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 17:12:52 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_directories import T_list, quantities_dir

### !!! .NPY DESCRIPTION !!! ###
# obs = (32, 5, 12)
# ind1: temperatures
# ind2: [MC, RG, SR continuous, SR rounded, SR sampled]
# ind3: [Mag, En, Susc, specHeat, Mag2, Mag4, En2, tpf(L/2), tpf(L/4), S0, S1, S2]

NAME = 'Simple2D16relu_L2_64_32_K513_PBC__MReg0.00EReg0.00fB1000'
NAME = 'Simple2D16relu_L3_64_16_16_K3333_PBC_MReg0.00EReg0.20B1000'
#NAME = 'Simple2D16relu_L2_64_32_K555_PBC_MReg0.00EReg0.30B1000'

# Load data
obs = np.load('%s/%s.npy'%(quantities_dir, NAME))

# Use rounding instead of sampling for the three lowest temperatures to correct Cv
obs[:3, -1] = obs[:3, -2]

def plot_quantity(q=0, figsize=(8, 5), L=16):
    # q: which quantity to plot
    plt.figure(figsize=figsize)
    plt.plot(T_list, obs[:, 0, q], color='blue', label='%dx%d MC'%(L, L))
    plt.plot(T_list, obs[:, 1, q], '--', color='green', label='%dx%d RG'%(L//2, L//2))
    plt.plot(T_list, obs[:, -1, q], 'x', color='red', label='%dx%d SR'%(L, L))
    
    plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle='--', color='k')
    plt.legend()
    
    plt.show()
    
def plot_four(figsize=(14, 8), L=16):
    # plots the four plots (M, E, chi, Cv)
    plt.figure(figsize=figsize)
    ylab = ['$M$', '$E$', '$\chi $', '$C_V$']
    for q in range(4):
        plt.subplot(221 + q)
        plt.plot(T_list, obs[:, 0, q], color='blue', label='%dx%d MC'%(L, L))
        plt.plot(T_list, obs[:, 1, q], '--', color='green', label='%dx%d RG'%(L//2, L//2))
        plt.plot(T_list, obs[:, -1, q], 'x', color='red', label='%dx%d SR'%(L, L))
        plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle='--', color='k')
        if q == 1:
            plt.legend()
        plt.xlabel('$T$')
        plt.ylabel(ylab[q])
    
    plt.show()