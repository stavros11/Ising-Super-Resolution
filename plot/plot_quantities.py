# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 17:12:52 2018

@author: Stavros
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from plot_directories import T_list, quantities_dir
# Use this T_list when plot_directories module is not available
#T_list = np.linspace(0.01, 4.538, 32)


### !!! .NPY DESCRIPTION !!! ###
# obs = (32, 5, 12)
# ind1: temperatures
# ind2: [MC, RG, SR continuous, SR rounded, SR sampled]
# ind3: [Mag, En, Susc, specHeat, Mag2, Mag4, En2, tpf(L/2), 
#        tpf(L/4), S0, S1, S2]

# Load data (fix .npy directory here)
NAME = 'Simple2D16relu_L3_64_16_16_K3333_MReg0.10EReg0.30_OLD'
obs = np.load('%s/%s.npy'%(quantities_dir, NAME))

# Use rounding instead of sampling for the five lowest temperatures 
# to correct noise in susc and Cv
obs[:5, -1] = obs[:5, -2]

def plot_one(q=0, figsize=(8, 5), L=16):
    # q: which quantity to plot
    plt.figure(figsize=figsize)
    plt.plot(T_list, obs[:, 0, q], color='blue', label='%dx%d MC'%(L, L))
    plt.plot(T_list, obs[:, 1, q], '--', color='green', label='%dx%d RG'%(L//2, L//2))
    plt.plot(T_list, obs[:, -1, q], 'x', color='red', label='%dx%d SR'%(L, L))
    
    plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle='--', color='k')
    plt.legend()
    
    plt.show()
    
def plot_four(figsize=(14, 8), L=16, save=False):
    # plots the four plots (M, E, chi, Cv)
    plt.figure(figsize=figsize)
    ylab = ['$M$', '$E$', '$\chi $', '$C_V$']
    text = ['a','b','c','d']
    
    tx = [0.2, 0.2, 0.2, 0.2]
    ty = [0.9, -0.45, 5.4, 1.55]
    
    for q in range(4):
        plt.subplot(221 + q)
        plt.plot(T_list, obs[:, 0, q], color='blue', label=''.join([r'%d'%L, r'$\times$', r'%d'%L, r' MC']),
                 linewidth=3.5, marker='o', markersize=11)
        plt.plot(T_list, obs[:, 1, q], color='blue', 
                 label=''.join([r'%d'%(L//2), r'$\times$', r'%d'%(L//2), r' RG']), linewidth=3.5, alpha=0.4,
                 marker='s', markersize=8)
        
        plt.plot(T_list, obs[:, -1, q], linestyle='--', color='red', label=''.join([r'%d'%L, r'$\times$', r'%d'%L, r' SR']),
                 linewidth=3.5, markersize=15, marker='d', alpha=0.7)
        
        plt.text(tx[q], ty[q], text[q], horizontalalignment='center', verticalalignment='center', 
                 fontweight='bold', fontsize=62)
        
        if q < 2:
            plt.locator_params(axis='y', nbins=6)
        else:
            plt.locator_params(axis='y', nbins=6)
        
        plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle='--', color='k', linewidth=1.8)
        plt.xlim([0, 4.6])
        
        if q == 0:
            plt.legend(loc='lower left', fontsize=50)
        plt.xlabel('$T$', fontsize=52)
        plt.ylabel(ylab[q], fontsize=52)
        
    
    if save:
        plt.subplots_adjust(left=0.1, right=0.95, top=0.96, bottom=0.08)
        #plt.savefig('%s.pdf'%NAME)
        plt.savefig('ups_RG2D.pdf')
    else:
        plt.show()

matplotlib.rcParams.update({'font.size': 42})
plot_four(figsize=(32, 20), save=True)
#plot_four(figsize=(12, 8), save=True)
