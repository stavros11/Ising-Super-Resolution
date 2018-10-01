# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 17:12:52 2018

@author: Stavros
"""

import numpy as np
import seaborn as sns
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
L = 16
NAME = 'Simple2D16relu_L3_32_32_16_K5533_PBC_MReg0.00EReg2.00B1000_corr'
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
        plt.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.1)
        #plt.savefig('%s.pdf'%NAME)
        plt.savefig('ups_RG2D_%s.pdf'%NAME)
    else:
        plt.show()

from mpl_toolkits.axes_grid.inset_locator import inset_axes

matplotlib.rcParams.update({'font.size': 38})
label_size = 46
text_size = 50

fig = plt.figure(figsize=(30, 7))
# set height ratios for sublots

cp = sns.color_palette("Paired", 10)

# the fisrt subplot
ax0 = plt.subplot(121)
line_mcM, = ax0.plot(T_list, obs[:, 0, 0], color=cp[1], alpha=0.8, linewidth=3.5, marker='')
line_rgM, = ax0.plot(T_list, obs[:, 1, 0], color=cp[0], linewidth=3.5, marker='', linestyle='--')
line_srM, = ax0.plot(T_list, obs[:, -1, 0], linestyle=' ', 
                     color=cp[4], marker='o', markersize=12)
plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle='--', color='k', linewidth=1.5)
plt.ylabel('$M$', fontsize=label_size)
plt.xlabel('$T$', fontsize=label_size)

plt.text(-0.78, 0.99, '(a)', horizontalalignment='center', verticalalignment='center', 
         fontsize=text_size)

if False:
    pass
else:
    ax_ins = inset_axes(ax0, 
                        width="30%", # width = 30% of parent_bbox
                        height="40%", # height : 1 inch
                        loc=1)
    plt.plot(T_list, obs[:, 0, 2], color=cp[3], alpha=0.8, linewidth=3)
    plt.plot(T_list, obs[:, 1, 2], color=cp[2], linewidth=3, linestyle='--')
    plt.plot(T_list, obs[:, -1, 2], linestyle=' ', 
                         color=cp[-1], alpha=0.8, marker='o', markersize=10)
    plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle=(0, (5, 1)), color='k', linewidth=1.5)
    plt.locator_params(axis='x', nbins=2)
    plt.locator_params(axis='y', nbins=3)
    plt.xlim([1.5, 3])
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.ylabel('$\chi $', fontsize=label_size - 10)
    plt.xlabel('$T$', fontsize=label_size - 10)


#the second subplot
ax1 = plt.subplot(122)
line_mcE, = ax1.plot(T_list, obs[:, 0, 1], color=cp[1], alpha=0.8, linewidth=3.5, marker='')
line_rgE, = ax1.plot(T_list, obs[:, 1, 1], color=cp[0], linewidth=3.5, marker='', linestyle='--')
line_srE, = ax1.plot(T_list, obs[:, -1, 1], linestyle=' ', 
                     color=cp[4], marker='o', markersize=12)
plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle=(0, (5, 1)), color='k', linewidth=1.5)
plt.ylabel('$E$', fontsize=label_size)
plt.xlabel('$T$', fontsize=label_size)

ax0.locator_params(axis='y', nbins=5)
ax1.locator_params(axis='y', nbins=5)

plt.text(-0.525, -0.25, '(b)', horizontalalignment='center', verticalalignment='center', 
         fontsize=text_size)


if False:
    pass
else:
    ax_ins2 = inset_axes(ax1, 
                        width="30%", # width = 30% of parent_bbox
                        height="40%", # height : 1 inch
                        loc=4)
    plt.plot(T_list, obs[:, 0, 3], color=cp[3], alpha=0.8, linewidth=3)
    plt.plot(T_list, obs[:, 1, 3], color=cp[2], linewidth=3, linestyle='--')
    plt.plot(T_list, obs[:, -1, 3], linestyle=' ', 
                         color=cp[-1], alpha=0.8, marker='o', markersize=10)
    plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle=(0, (5, 1)), color='k', linewidth=1.5)
    plt.locator_params(axis='x', nbins=2)
    plt.locator_params(axis='y', nbins=3)
    plt.xlim([1.5, 3])
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.ylabel('$C_V$', fontsize=label_size - 10)
    plt.xlabel('$T$', fontsize=label_size - 10)
    ax_ins2.xaxis.set_label_position('top') 
    ax_ins2.xaxis.tick_top()
    

# put legend on first subplot
ax0.legend((line_mcM, line_rgM, line_srM), (
        ''.join([r'%d'%L, r'$\times$', r'%d'%L, r' MC']), 
        ''.join([r'%d'%(L//2), r'$\times$', r'%d'%(L//2), r' DS']),
        ''.join([r'%d'%L, r'$\times$', r'%d'%L, r' SR'])), loc='lower left', fontsize=37)

# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)

#plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.10)
plt.savefig('test2D_pure_%s.pdf'%NAME, bbox_inches='tight')
