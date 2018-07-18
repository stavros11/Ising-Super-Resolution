# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:33:22 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 32})

from plot_directories import quantities_dir1D_fixed
# Note T_list is different for 1D
T_list = np.linspace(0.01, 3.515, 32)
# Renormalized temperature
T_ren = 2.0 / np.arccosh(np.exp(2.0 / T_list))

### !!! .NPY DESCRIPTION !!! ###
# obs = (32, 5, 12)
# ind1: temperatures
# ind2: [MC, RG, SR continuous, SR rounded, SR sampled]
# ind3: [Mag, En, Susc, specHeat]

# Load data (fix .npy directory here)
NAME = 'Regular1D32relu_L1_20_MReg0.10EReg0.30B1000_gaussian_sampling'
obs = np.load('%s/%s.npy'%(quantities_dir1D_fixed, NAME))

# Use rounding instead of sampling for the three lowest temperatures 
# to correct noise in susc and Cv
    
def plot_four(figsize=(10,6), save=False):
    color_list = ['blue', 'green', 'red']
    shape_list = ['-', '--', 'x', '*']
    labels = ['32 MC', '16 MC', '32 SR fixed']
    qlabels = ['$M$', '$E$', '$\chi $', '$C_V$']
    Tl = [T_list, T_list, T_ren]
    # select sampling
    select = [0, 1, 4]
    
    plt.figure(figsize=figsize)
    for q in range(4):
        plt.subplot(221+q)
        for i in range(3):
            plt.plot(Tl[i], obs[:, select[i], q], shape_list[i],
                     color=color_list[i], label=labels[i])
        plt.xlabel('$T$')
        plt.ylabel(qlabels[q])
        if q == 1:
            plt.legend()
            
    if save:
        plt.savefig('%s_Fixed_Four.pdf'%NAME)
    else:
        plt.show()
        

def plot_two(figsize=(15,6), save=False, linewidth=1.5):
    color_list = ['blue', 'green', 'red']
    shape_list = ['-', '--', 'x', '*']
    labels = ['32 MC', '16 MC', '32 SR fixed']
    qlabels = ['$M$', '$E$', '$\chi $', '$C_V$']
    Tl = [T_list, T_list, T_ren]
    # select sampling
    select = [0, 1, 4]
    
    plt.figure(figsize=figsize)
    for q in range(2):
        plt.subplot(121+q)
        for i in range(3):
            plt.plot(Tl[i], obs[:, select[i], q], shape_list[i],
                     color=color_list[i], label=labels[i], linewidth=linewidth)
        plt.xlabel('$T$')
        plt.ylabel(qlabels[q])
        if q == 1:
            plt.legend()
            
    if save:
        plt.savefig('%s_Fixed_Two.pdf'%NAME)
    else:
        plt.show()