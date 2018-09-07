# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 22:23:40 2018

@author: Stavros
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 42})

from plot_directories import T_list1D as T_list
from plot_directories import quantities_dir1D, quantities_dir1D_fixed
# Use this T_list when plot_directories module is not available
#T_list = np.linspace(0.01, 4.538, 32)


### !!! .NPY DESCRIPTION !!! ###
# obs = (32, 5, 12)
# ind1: temperatures
# ind2: [MC, RG, SR continuous, SR rounded, SR sampled]
# ind3: [Mag, En, Susc, specHeat, Mag2, Mag4, En2, tpf(L/2), 
#        tpf(L/4), S0, S1, S2]

# Load data (fix .npy directory here)
NAME = 'Simple1D32relu_L1_32_K53_PBC_MReg0.00EReg0.00B1000'
obs = np.load('%s/%s.npy'%(quantities_dir1D, NAME))
fixed_older = np.load('%s/%s.npy'%(quantities_dir1D_fixed, NAME))

# Use rounding instead of sampling for the five lowest temperatures 
# to correct noise in susc and Cv
obs[:3, -1] = obs[:3, -2]
fixed_older[:6, -1] = fixed_older[:6, -2]

def plot_one(q=0, figsize=(8, 5), L=32):
    # q: which quantity to plot
    plt.figure(figsize=figsize)
    plt.plot(T_list, obs[:, 0, q], color='blue', label='%d MC'%L)
    plt.plot(T_list, obs[:, 1, q], '--', color='green', label='%d RG'%(L//2))
    plt.plot(T_list, obs[:, -1, q], 'x', color='red', label='%d SR'%L)
    
    plt.legend()
    
    plt.show()
    
def plot_four(qtot=4, figsize=(14, 8), L=32, save=False):
    # plots the four plots (M, E, chi, Cv)
    plt.figure(figsize=figsize)
    ylab = ['$M$', '$E$', '$\chi $', '$C_V$']
    for q in range(qtot):
        plt.subplot(221 + q)
        plt.plot(T_list, obs[:, 0, q], color='#ff0000', label='$N=$%d MC'%L,
                 linewidth=2.0)
        plt.plot(T_list, obs[:, 1, q], color='#e66464', 
                 label='$N=$%d RG'%(L//2), linewidth=2.0, alpha=0.6)
        
        plt.plot(T_list, obs[:, -1, q], 'o--', color='blue', label='$N=$%d SR'%L, linewidth=1.2)
        if q == 0:
            plt.legend()
        plt.xlabel('$T$')
        plt.ylabel(ylab[q])
        
        plt.locator_params(axis='y', nbins=5)
    
    if save:
        plt.savefig('%s.pdf'%NAME)
    else:
        plt.show()
    
    
    
matplotlib.rcParams.update({'font.size': 38})
test_size = 50 #text
label_size = 46
        
#fig = plt.figure(figsize=(30, 10))
## set height ratios for sublots
#gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1]) 
#
## the fisrt subplot
#ax0 = plt.subplot(gs[0])
#line_mcM, = ax0.plot(T_list, obs[:, 0, 0], color='blue', linewidth=3.5, alpha=0.9, marker='o', markersize=11)
#line_rgM, = ax0.plot(T_list, obs[:, 1, 0], color='blue', linewidth=3.5, alpha=0.4, marker='s', markersize=8)
#line_srM, = ax0.plot(T_list, obs[:, -1, 0], linestyle='--', color='red', linewidth=3.5, 
#                     alpha=0.8, markersize=15, marker='d')
#plt.ylabel('$M$', fontsize=label_size)
#ax0.yaxis.set_label_coords(0.06,0.5)
#
##the second subplot
#ax1 = plt.subplot(gs[2], sharex = ax0)
#line_mcE, = ax1.plot(T_list, obs[:, 0, 1], color='blue', linewidth=3.5, alpha=0.9, marker='o', markersize=11)
#line_rgE, = ax1.plot(T_list, obs[:, 1, 1], color='blue', linewidth=3.5, alpha=0.4, marker='s', markersize=8)
#line_srE, = ax1.plot(T_list, obs[:, -1, 1], 'o--', color='red', linewidth=3.5, 
#                     alpha=0.8, markersize=15, marker='d')
#plt.setp(ax0.get_xticklabels(), visible=False)
#plt.ylabel('$E$', fontsize=label_size)
#ax1.yaxis.set_label_coords(0.06,0.5)
#
#ax0.locator_params(axis='y', nbins=5)
#ax1.locator_params(axis='y', nbins=5)
## remove last tick label for the second subplot
#yticks = ax0.yaxis.get_major_ticks()
#yticks[0].label1.set_visible(False)
#
## put lened on first subplot
#ax0.legend((line_mcM, line_rgM, line_srM), ('$N=32$ MC', '$N=16$ RG', '$N=32$ SR'), 
#           loc='upper right', fontsize=legend_size)
#
#plt.text(0.15, 0.8, 'a', horizontalalignment='center', verticalalignment='center', 
#                 fontweight='bold', fontsize=test_size)
#plt.text(0.15, -0.15, 'b', horizontalalignment='center', verticalalignment='center', 
#                 fontweight='bold', fontsize=test_size)
#
#plt.xlabel('$T$', fontsize=label_size)
#plt.locator_params(axis='x', nbins=5)
#plt.xlim([0, 3.7])
## remove vertical gap between subplots
#plt.subplots_adjust(hspace=.0)
#
#
## the fisrt subplot
#T_ren = 2.0 / np.arccosh(np.exp(2.0 / T_list))
#ax2 = plt.subplot(gs[1])
#line_mcMf, = ax2.plot(T_list, fixed_older[:, 0, 0], color='blue', linewidth=3.5, alpha=0.9, marker='o', markersize=11)
#line_rgMf, = ax2.plot(T_list, fixed_older[:, 1, 0], color='blue', linewidth=3.5, alpha=0.4, marker='s', markersize=8)
#line_srMf, = ax2.plot(T_ren, fixed_older[:, -1, 0], 'o--', color='red', linewidth=3.5, 
#                     alpha=0.8, markersize=15, marker='d')
#plt.ylabel('$M$', fontsize=label_size)
#ax2.yaxis.set_label_coords(0.06,0.5)
#
##the second subplot
#ax3 = plt.subplot(gs[3], sharex = ax2)
#line_mcEf, = ax3.plot(T_list, fixed_older[:, 0, 1], color='blue', linewidth=3.5, alpha=0.9, marker='o', markersize=11)
#line_rgEf, = ax3.plot(T_list, fixed_older[:, 1, 1], color='blue', linewidth=3.5, alpha=0.4, marker='s', markersize=8)
#line_srEf, = ax3.plot(T_ren, fixed_older[:, -1, 1], 'o--', color='red', linewidth=3.5, 
#                     alpha=0.8, markersize=15, marker='d')
#plt.setp(ax2.get_xticklabels(), visible=False)
#plt.ylabel('$E$', fontsize=label_size)
#ax3.yaxis.set_label_coords(0.06,0.5)
#
#ax2.locator_params(axis='y', nbins=5)
#ax3.locator_params(axis='y', nbins=5)
## remove last tick label for the second subplot
#yticks = ax2.yaxis.get_major_ticks()
#yticks[0].label1.set_visible(False)
#
#plt.text(0.15, 0.4, 'c', horizontalalignment='center', verticalalignment='center', 
#                 fontweight='bold', fontsize=test_size)
#plt.text(0.15, -0.35, 'd', horizontalalignment='center', verticalalignment='center', 
#                 fontweight='bold', fontsize=test_size)
#
## put lened on first subplot
#ax2.legend((line_mcMf, line_rgMf, line_srMf), ('$N=32$ MC', '$N=16$ MC', '$N=32$ SR'), 
#           loc='upper right', fontsize=legend_size)
#
#plt.xlabel('$T$', fontsize=label_size)
#plt.xlim([0, 3.7])
#plt.locator_params(axis='x', nbins=5)
## remove vertical gap between subplots
#plt.subplots_adjust(hspace=.0)
#
##plt.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.15)
from mpl_toolkits.axes_grid.inset_locator import inset_axes

T_ren = 2.0 / np.arccosh(np.exp(2.0 / T_list))
def get_errors():
    errors = np.zeros([len(T_list), fixed_older.shape[-1]])
    for (iT, T) in enumerate(T_ren):
        dif = np.abs(T - T_list)
        i1 = dif.argmin()
        dif[i1] = 1000
        i2 = dif.argmin()
        mc_values = (fixed_older[i2, 0] - fixed_older[i1, 0]) * (T - T_list[i1])/ (T_list[i2] - T_list[i1])
        mc_values+= fixed_older[i1, 0]
        
        errors[iT] = np.abs((mc_values - fixed_older[iT, -1]) / mc_values)
        
    return errors
        
errors = get_errors()

fig = plt.figure(figsize=(30, 7))
#gs = gridspec.GridSpec(1, 2)
#cmap = plt.cm.get_cmap('Paired_r', lut=4)
cp = sns.color_palette("Paired")

ax2 = fig.add_subplot(121)
line_mcMf, = plt.plot(T_list, fixed_older[:, 0, 0], color=cp[1], alpha=0.8, linewidth=3.5)
line_rgMf, = plt.plot(T_list, fixed_older[:, 1, 0], color=cp[0], linewidth=3.5, linestyle='--')
line_srMf, = plt.plot(T_ren, fixed_older[:, -1, 0], color=cp[4], linestyle='', markersize=12, marker='o')
plt.ylabel('$M$', fontsize=label_size)
plt.xlabel('$T$', fontsize=label_size)

plt.text(0, 0.97, 'a', horizontalalignment='center', verticalalignment='center', 
         fontweight='bold', fontsize=test_size)

ax_ins = inset_axes(ax2, 
                    width="40%", # width = 30% of parent_bbox
                    height="40%", # height : 1 inch
                    loc=1)
plt.plot(T_list, 100*errors[:, 0], linestyle=':', linewidth=2.0, color=cp[3], alpha=0.7,
         markersize=12, marker='^')
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=3)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.ylabel('Error (%)', fontsize=label_size - 14)
plt.xlabel('$T$', fontsize=label_size - 14)


ax3 = fig.add_subplot(122)
line_mcEf, = ax3.plot(T_list, fixed_older[:, 0, 1], color=cp[1], alpha=0.8, linewidth=3.5)
line_rgEf, = ax3.plot(T_list, fixed_older[:, 1, 1], color=cp[0], linewidth=3.5, linestyle='--')
line_srEf, = ax3.plot(T_ren, fixed_older[:, -1, 1], color=cp[4], linestyle='', markersize=12, marker='o')
plt.ylabel('$E$', fontsize=label_size)
plt.xlabel('$T$', fontsize=label_size)

ax3.text(0, -0.32, 'b', horizontalalignment='center', verticalalignment='center', 
         fontweight='bold', fontsize=test_size)

ax_ins2 = inset_axes(ax3, 
                    width="40%", # width = 30% of parent_bbox
                    height="40%", # height : 1 inch
                    loc=4)
plt.plot(T_list, 100*errors[:, 1], linestyle=':', linewidth=2.0, color=cp[3], alpha=0.7,
         markersize=15, marker='^')
ax_ins2.xaxis.tick_top()
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=3)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.ylabel('Error (%)', fontsize=label_size - 14)
plt.xlabel('$T$', fontsize=label_size - 14)
ax_ins2.xaxis.set_label_position('top') 


ax2.legend((line_mcMf, line_rgMf, line_srMf), ('$N=32$ MC', '$N=16$ MC', '$N=32$ SR'), 
           loc='lower left', fontsize=34)

plt.savefig('ups_real1D.pdf', bbox_inches='tight')