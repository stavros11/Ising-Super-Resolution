# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 17:20:57 2018

@author: User
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from os import getcwd, path

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from plot_directories import T_list, quantities_real_dir
# Use this T_list when plot_directories module is not available
#T_list = np.linspace(0.01, 4.538, 32)

#def inv_curve(x, a, b):
#    return b / np.arccosh(np.exp(a / x))

## Found from magnetization: RG^(-1)(MC(T)) ##
cut_iT = 5
T_ren_inv = np.array([0., 0., 0., 0., 0.,
       1.21835191, 1.22976684, 1.39674347, 1.51484435, 1.65761354,
       1.75902208, 1.85837041, 1.95260925, 2.07132396, 2.13716533,
       2.25437054, 2.29606717, 2.38018868, 2.44845189, 2.51316151,
       2.58725426, 2.6448879 , 2.7110948 , 2.74426717, 2.81525268,
       2.87031377, 2.90806294, 2.98742994, 3.03780331, 3.10501399,
       3.17323991, 3.19663683])

### !!! .NPY DESCRIPTION !!! ###
# obs = (32, 5, 12)
# ind1: temperatures
# ind2: [MC, RG, SR continuous, SR rounded, SR sampled]
# ind3: [Mag, En, Susc, specHeat, Mag2, Mag4, En2, tpf(L/2), 
#        tpf(L/4), S0, S1, S2]

# Load renormalization parameters (fix .npy directories)
git_folder = path.split(getcwd())[0]
parameters_dir = path.join(git_folder, 'renormalization')
a, b = np.zeros(2), np.zeros(2)
a[0], b[0] = np.load(path.join(parameters_dir, 'Magnetization_Transformation_Params_L16.npy'))
a[1], b[1] = np.load(path.join(parameters_dir, 'Energy_Transformation_Params_L16.npy'))

# Load data (fix .npy directory here)
L = 16
NAME = 'Simple2D16relu_L3_64_16_16_K3333_PBC_MReg0.00EReg0.20B1000'
obs = np.load('%s/%s.npy'%(quantities_real_dir, NAME))

# Use rounding instead of sampling for the five lowest temperatures 
# to correct noise in susc and Cv
obs[:4, -1] = obs[:4, -2]

def plot_one(q=0, figsize=(8, 5), L=16):
    # q: which quantity to plot
    plt.figure(figsize=figsize)
    plt.plot(T_list, obs[:, 0, q], color='blue', label='%dx%d MC'%(L, L))
    plt.plot(T_list, obs[:, 1, q], '--', color='green', label='%dx%d RG'%(L//2, L//2))
    plt.plot(T_ren_inv[cut_iT:], obs[cut_iT:, -1, q], 
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
        
        plt.plot(T_ren_inv[cut_iT:], obs[cut_iT:, -1, q],
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
        plt.plot(T_ren_inv[cut_iT:], obs[cut_iT:, -1, q],
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
line_srM, = ax0.plot(T_ren_inv[cut_iT:], obs[cut_iT:, -1, 0], linestyle=' ', 
                     color=cp[4], marker='o', markersize=12)
plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle='--', color='k', linewidth=1.5)
plt.ylabel('$M$', fontsize=label_size)
plt.xlabel('$T$', fontsize=label_size)

plt.text(0.0, 0.975, 'a', horizontalalignment='center', verticalalignment='center', 
                 fontweight='bold', fontsize=text_size)

ax_ins = inset_axes(ax0, 
                    width="30%", # width = 30% of parent_bbox
                    height="40%", # height : 1 inch
                    loc=1)
plt.plot(T_list, obs[:, 0, 2], color=cp[3], alpha=0.8, linewidth=3)
plt.plot(T_list, obs[:, 1, 2], color=cp[2], linewidth=3, linestyle='--')
plt.plot(T_ren_inv[cut_iT:], obs[cut_iT:, -1, 2], linestyle=' ', 
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
line_srE, = ax1.plot(T_ren_inv[cut_iT:], obs[cut_iT:, -1, 1], linestyle=' ', 
                     color=cp[4], marker='o', markersize=12)
plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle=(0, (5, 1)), color='k', linewidth=1.5)
plt.ylabel('$E$', fontsize=label_size)
plt.xlabel('$T$', fontsize=label_size)

ax0.locator_params(axis='y', nbins=5)
ax1.locator_params(axis='y', nbins=5)

plt.text(0.0, -0.545, 'b', horizontalalignment='center', verticalalignment='center', 
                 fontweight='bold', fontsize=text_size)

ax_ins2 = inset_axes(ax1, 
                    width="30%", # width = 30% of parent_bbox
                    height="40%", # height : 1 inch
                    loc=4)
plt.plot(T_list, obs[:, 0, 3], color=cp[3], alpha=0.8, linewidth=3)
plt.plot(T_list, obs[:, 1, 3], color=cp[2], linewidth=3, linestyle='--')
plt.plot(T_ren_inv[cut_iT:], obs[cut_iT:, -1, 3], linestyle=' ', 
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
        ''.join([r'%d'%(L//2), r'$\times$', r'%d'%(L//2), r' MC']),
        ''.join([r'%d'%L, r'$\times$', r'%d'%L, r' SR'])), loc='lower left', fontsize=37)

# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)

#plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.10)
plt.savefig('ups_real2D_%s.pdf'%NAME, bbox_inches='tight')


#plt.figure(figsize=(8, 6))
#plt.plot(T_list, obs[:, 0, 3], color='blue', linewidth=3.5, marker='', markersize=11)
#plt.plot(T_list, obs[:, 1, 3], color='blue', linewidth=3.5, alpha=0.4, marker='', markersize=8)
#plt.plot(inv_curve(T_list, a=a[1], b=b[1]), obs[:, -1, 3], linestyle=' ', color='red',
#                     marker='o', markersize=15, alpha=0.8)
#plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle=(0, (5, 1)), color='k', linewidth=1.5)

