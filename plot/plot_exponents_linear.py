# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:36:30 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib import rcParams
rcParams.update({'font.size': 32})

# If plot_directories module is available:
from plot_directories import multiple_exponents_dir
# otherwise fix directory

### !!! .NPY DESCRIPTION !!! ###
# obs = (7, Upsamplings+1)
# ind1: [mag, en, susc, Cv, mag^2, mag^4, en^2, binder, ksiA, ksiB, 
#       tpf(L/2), tpf(L/4)]
# ind2: different lengths

# Load data (fix .npy directory here!)
NAME = 'Simple2D16relu_L2_64_16_16_K3333_C1UP5_OLD'
obs = np.load('%s/%s.npy'%(multiple_exponents_dir, NAME))

n_obs, upsamplings = obs.shape
upsamplings += -1
L_list = 2**np.arange(4, upsamplings+5)

print(linregress(np.log10(L_list), np.log10(obs[2])))
print(linregress(np.log10(L_list/2), np.log10(obs[10])))
print(linregress(np.log10(L_list/4), np.log10(obs[11])))

def plot_one(q=2, figsize=(8,5), save=False):
    # q=2 for susceptibility
    plt.figure(figsize=figsize)
    plt.loglog(L_list, obs[q], '-o', color='blue', linewidth=2.0)
    plt.xlabel('$L$')
    plt.ylabel('$\chi $')
    plt.grid(which='major')
    plt.grid(which='minor')
    
    if save:
        plt.savefig('%s_susc.pdf'%NAME)
    else:
        plt.show()
        
def plot_TPF(figsize=(8,5), save=False):
    plt.figure(figsize=figsize)
    plt.loglog(L_list/2, obs[10], '-o', color='red', 
               linewidth=2.0, label='$L/2$')
    plt.loglog(L_list/4, obs[11], '-o', color='green', 
               linewidth=2.0, label='$L/4$')
    plt.xlabel('$r$')
    plt.ylabel('Two Point Function')
    plt.grid(which='major')
    plt.grid(which='minor')
    plt.legend()
    
    if save:
        plt.savefig('%s_tpf.pdf'%NAME)
    else:
        plt.show()
    