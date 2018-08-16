# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 17:20:57 2018

@author: User
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 62})

from plot_directories import T_list, quantities_real_dir
# Use this T_list when plot_directories module is not available
#T_list = np.linspace(0.01, 4.538, 32)

def inv_curve(x, a, b):
    return b / np.arccosh(np.exp(a / x))

### !!! .NPY DESCRIPTION !!! ###
# obs = (32, 5, 12)
# ind1: temperatures
# ind2: [MC, RG, SR continuous, SR rounded, SR sampled]
# ind3: [Mag, En, Susc, specHeat, Mag2, Mag4, En2, tpf(L/2), 
#        tpf(L/4), S0, S1, S2]

# Load renormalization parameters (fix .npy directories)
parameters_dir = 'C:/Users/User/Documents/Stavros/Ising-Super-Resolution/renormalization'
a, b = np.zeros(2), np.zeros(2)
a[0], b[0] = np.load('%s/Magnetization_Transformation_Params_L16.npy'%parameters_dir)
a[1], b[1] = np.load('%s/Energy_Transformation_Params_L16.npy'%parameters_dir)

# Load data (fix .npy directory here)
NAME = 'Simple2D16relu_L3_64_16_16_K3333_MReg0.10EReg0.30_OLD'
obs = np.load('%s/%s.npy'%(quantities_real_dir, NAME))

# Use rounding instead of sampling for the five lowest temperatures 
# to correct noise in susc and Cv
obs[:4, -1] = obs[:4, -2]

def plot_one(q=0, figsize=(8, 5), L=16):
    # q: which quantity to plot
    plt.figure(figsize=figsize)
    plt.plot(T_list, obs[:, 0, q], color='blue', label='%dx%d MC'%(L, L))
    plt.plot(T_list, obs[:, 1, q], '--', color='green', label='%dx%d RG'%(L//2, L//2))
    plt.plot(inv_curve(T_list, a=a[q], b=b[q]), obs[:, -1, q], 
             'x', color='red', label='%dx%d SR'%(L, L))
    
    plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle='--', color='k')
    plt.legend()
    
    plt.show()
    
def plot_two(figsize=(18, 6), L=16, linewidth=1.5, save=False):
    # plots the four plots (M, E)
    plt.figure(figsize=figsize)
    ylab = ['$M$', '$E$']
    for q in range(2):
        plt.subplot(121 + q)
        plt.plot(T_list, obs[:, 0, q], color='blue', label='%dx%d MC'%(L, L),
                 linewidth=linewidth)
        plt.plot(T_list, obs[:, 1, q], '--', color='green', 
                 label='%dx%d MC'%(L//2, L//2), linewidth=linewidth)
        
        plt.plot(inv_curve(T_list, a=a[q], b=b[q]), obs[:, -1, q], 
                 '*', color='red', label='%dx%d SR'%(L, L))
        plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle='--', color='k')
        if q == 1:
            plt.legend()
        plt.xlabel('$T$')
        plt.ylabel(ylab[q])
    
    if save:
        plt.savefig('%s.pdf'%NAME)
    else:
        plt.show()
        
def plot_two_unfixed(figsize=(18, 6), L=16, linewidth=1.5, save=False):
    # plots the four plots (M, E)
    plt.figure(figsize=figsize)
    ylab = ['$M$', '$E$']
    for q in range(2):
        plt.subplot(121 + q)
        plt.plot(T_list, obs[:, 0, q], color='blue', label='%dx%d MC'%(L, L),
                 linewidth=linewidth)
        plt.plot(T_list, obs[:, 1, q], '--', color='green', 
                 label='%dx%d MC'%(L//2, L//2), linewidth=linewidth)
        
        plt.plot(T_list, obs[:, -1, q], 'x', color='black', label='%dx%d SR'%(L, L))
        plt.plot(inv_curve(T_list, a=a[q], b=b[q]), obs[:, -1, q], 
                 '*', color='red', label='%dx%d SR fixed'%(L, L))
        plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle='--', color='k')
        if q == 1:
            plt.legend()
        plt.xlabel('$T$')
        plt.ylabel(ylab[q])
    
    if save:
        plt.savefig('%s.pdf'%NAME)
    else:
        plt.show()
   
fig = plt.figure(figsize=(20, 20))
# set height ratios for sublots
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 

# the fisrt subplot
ax0 = plt.subplot(gs[0])
line_mcM, = ax0.plot(T_list, obs[:, 0, 0], color='blue', linewidth=3.5)
line_rgM, = ax0.plot(T_list, obs[:, 1, 0], color='blue', linewidth=3.5, alpha=0.3)
line_srM, = ax0.plot(inv_curve(T_list, a=a[0], b=b[0]), obs[:, -1, 0], 'o--', color='red', linewidth=3.0,
                     markersize=10)
plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle='--', color='k', linewidth=1.5)
plt.ylabel('$M$')

#the second subplot
ax1 = plt.subplot(gs[1], sharex = ax0)
line_mcE, = ax1.plot(T_list, obs[:, 0, 1], color='blue', linewidth=2.5)
line_rgE, = ax1.plot(T_list, obs[:, 1, 1], color='blue', linewidth=2.5, alpha=0.3)
line_srE, = ax1.plot(inv_curve(T_list, a=a[0], b=b[0]), obs[:, -1, 1], 'o--', color='red', linewidth=3.0,
                     markersize=10)
plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle='--', color='k', linewidth=1.5)
plt.setp(ax0.get_xticklabels(), visible=False)
plt.ylabel('$E$')

ax0.locator_params(axis='y', nbins=5)
ax1.locator_params(axis='y', nbins=5)
# remove last tick label for the second subplot
yticks = ax0.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)

# put lened on first subplot
L = 16
ax0.legend((line_mcM, line_rgM, line_srM), (
        ''.join([r'%d'%L, r'$\times$', r'%d'%L, r' MC']), 
        ''.join([r'%d'%(L//2), r'$\times$', r'%d'%(L//2), r' MC']),
        ''.join([r'%d'%L, r'$\times$', r'%d'%L, r' MC'])), loc='lower left')

plt.xlabel('$T$')
# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
plt.savefig('ups_real2D.pdf')
