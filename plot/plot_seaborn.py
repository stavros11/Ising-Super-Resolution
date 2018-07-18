# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 12:32:37 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
from seaborn import distplot
from matplotlib import rcParams
rcParams.update({'font.size': 32})

from plot_directories import T_list, seaborn_dir
# Use this T_list when plot_directories module is not available
#T_list = np.linspace(0.01, 4.538, 32)

### !!! .NPY DESCRIPTION !!! ###
# obs = (32, 5, 2, 10000)
# ind1: temperatures
# ind2: [MC, RG, SR continuous, SR rounded, SR sampled]
# ind3: [Mag, En]
# ind4: different samples

# Load data (fix .npy directory here!)
NAME = 'Simple2D16relu_L3_64_16_16_K3333_MReg0.10EReg0.30B1000_OLD'
obs = np.load('%s/%s.npy'%(seaborn_dir, NAME))

def plot_one(iT, q=0, bins=20, figsize=(10, 6), save=False):
    # If iT < 5 select rounded instead of sampled
    if iT < 5:
        sampled = 3
    else:
        sampled = 4 
        
    # Select temperature, quantity and [MC, RG, SR sampled]
    obs_loc = obs[iT, np.array([0, 1, sampled]), q]
    
    color_list = ['blue', 'green', 'red']
    label_list = ['MC', 'RG', 'SR']
    alphas = [1.0, 0.75, 0.5]
    lb = ['Magnetization', 'Energy'][q]
    
    plt.figure(figsize=figsize)
    for i in range(3):
        sns_hist = distplot(obs_loc[i], bins=bins, color=color_list[i], 
                            label=label_list[i], 
                            norm_hist=True, 
                            hist_kws=dict(alpha=alphas[i]))
    plt.xlabel(lb)
    plt.ylabel('PDF')
    plt.suptitle('T = %.4f'%T_list[iT])
    plt.legend()
    
    if save:
        fig = sns_hist.get_figure()
        fig.savefig('OneHist_%s_T%.4f.pdf'%(NAME, T_list[iT]))
    else:
        plt.show()
        
def plot_two(iT, bins=20, figsize=(15, 5), save=False,
             tx=0.2, ty=5.5, text='a)'):
    # If iT < 5 select rounded instead of sampled
    if iT < 5:
        sampled = 3
    else:
        sampled = 4        
    
    # Select temperature and [MC, SR sampled]
    mag = obs[iT, np.array([0, 1, sampled]), 0]
    en  = obs[iT, np.array([0, 1, sampled]), 1]
    
    color_list = ['blue', 'green', 'red']
    label_list = ['MC', 'RG', 'SR']
    alphas = [1.0, 0.75, 0.5]
    
    plt.figure(figsize=figsize)
    plt.subplot(121)
    for i in range(3):
        distplot(mag[i], bins=bins, color=color_list[i], label=label_list[i],
                 norm_hist=True, hist_kws=dict(alpha=alphas[i]))
    plt.xlabel('Magnetization')
    plt.ylabel('PDF')
    plt.text(tx, ty, text, horizontalalignment='center', 
             verticalalignment='center', fontsize=50)
    
    plt.subplot(122)
    for i in range(3):
        distplot(en[i], bins=bins, color=color_list[i], label=label_list[i],
                 norm_hist=True, hist_kws=dict(alpha=alphas[i]))
    plt.xlabel('Energy')
    plt.ylabel('PDF')
    plt.legend()
    
    if save:
        plt.savefig('Seaborn_%s_T%.4f.pdf'%(NAME, T_list[iT]))
    else:
        plt.show()
  
def plot_two_temperatures(iT=(15, 20), bins=20, figsize=(15, 5), save=False, 
                          tx=(0.0, 0.95), ty=(5.0, 3.2), text=('a)', 'b)')):
    # If iT < 5 select rounded instead of sampled
    if iT < 5:
        sampled = 3
    else:
        sampled = 4
        
    color_list = ['blue', 'green', 'red']
    label_list = ['MC', 'RG', 'SR']
    alphas = [1.0, 0.75, 0.5]
    
    for i in range(2):
        # Select temperature and [MC, SR sampled]
        mag = obs[iT[i], np.array([0, 1, sampled]), 0]
        en  = obs[iT[i], np.array([0, 1, sampled]), 1]
            
        plt.figure(figsize=figsize)
        plt.subplot(141)
        for i in range(3):
            distplot(mag[i], bins=bins, color=color_list[i], label=label_list[i],
                     norm_hist=True, hist_kws=dict(alpha=alphas[i]))
        plt.xlabel('Magnetization')
        plt.ylabel('PDF')
        plt.text(tx[i], ty[i], text[i], horizontalalignment='center', 
                 verticalalignment='center', fontsize=50)
        
        plt.subplot(142)
        for i in range(3):
            distplot(en[i], bins=bins, color=color_list[i], label=label_list[i],
                     norm_hist=True, hist_kws=dict(alpha=alphas[i]))
        plt.xlabel('Energy')
        plt.ylabel('PDF')
        plt.legend()
    
    if save:
        plt.savefig('Seaborn_%s_T%.4f.pdf'%(NAME, T_list[iT]))
    else:
        plt.show()    
