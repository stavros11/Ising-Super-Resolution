# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:36:30 2018

@author: Stavros
"""

import numpy as np
from scipy.stats import linregress
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

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

#print(linregress(np.log10(L_list), np.log10(obs[2])))
#print(linregress(np.log10(L_list/2), np.log10(obs[10])))
#print(linregress(np.log10(L_list/4), np.log10(obs[11])))

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
        
        
matplotlib.rcParams.update({'font.size': 52})
label_font = 60
legend_font = 64
marker_size = 20
text_size = 70
        
#fig = plt.figure(figsize=(20, 20))
## set height ratios for sublots
#gs = gridspec.GridSpec(2, 2) 
#
## the fisrt subplot
#ax0 = plt.subplot(gs[0, 0])
#line0, = ax0.plot(np.log10(L_list), np.log10(obs[2]), linestyle='--', marker='d', markersize=marker_size, color='blue', linewidth=2.0)
#plt.xlabel('$\log \, L$', fontsize=label_font)
#ax0.xaxis.set_label_coords(0.5,0.12)
#ax0.yaxis.set_label_coords(0.13,0.5)
#plt.ylabel('$\log \, \chi $', fontsize=label_font)
#ax0.locator_params(axis='x', nbins=2)
#ax0.locator_params(axis='y', nbins=2)
#ax0.text(1.22, 2.7, 'a', horizontalalignment='center', verticalalignment='center', 
#                 fontweight='bold', fontsize=text_size)
#
#
## the second subplot
#ax1 = plt.subplot(gs[1, :])
#line1, = ax1.plot(np.log10(L_list/2), np.log10(obs[10]), linestyle='--', marker='s', markersize=marker_size, color='red', 
#                  linewidth=2.0, label='$r=L/2$')
#line11, = ax1.plot(np.log10(L_list/4), np.log10(obs[10]), linestyle='--', marker='^', markersize=marker_size, color='green', 
#                  linewidth=2.0, label='$r=L/4$')
#ax1.xaxis.set_label_coords(0.5,0.12)
#ax1.yaxis.set_label_coords(0.06,0.5)
#plt.xlabel('$\log \, r$', fontsize=label_font)
#plt.ylabel('$\log \, G(r) $', fontsize=label_font)
#plt.legend(loc='upper right', fontsize=legend_font)
#ax1.locator_params(axis='x', nbins=5)
#ax1.locator_params(axis='y', nbins=2)
#ax1.text(0.57, -0.28, 'c', horizontalalignment='center', verticalalignment='center', 
#                 fontweight='bold', fontsize=text_size)
#
## the third subplot
#ax2 = plt.subplot(gs[0, 1])
##ax2.set_xticklabels([])
##ax2.set_yticklabels([])
#line2, = ax2.plot(np.log10(L_list), np.log10(obs[0]), linestyle='--', marker='o', markersize=marker_size, color='magenta', linewidth=2.0)
#plt.xlabel('$\log \, L$', fontsize=label_font)
#plt.ylabel('$\log \, M $', fontsize=label_font)
#ax2.xaxis.set_label_coords(0.5,0.12)
#ax2.yaxis.set_label_coords(0.13,0.5)
#ax2.locator_params(axis='x', nbins=2)
#ax2.locator_params(axis='y', nbins=2)
#ax2.text(2.4, -0.151, 'b', horizontalalignment='center', verticalalignment='center', 
#                 fontweight='bold', fontsize=text_size)
#
#plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08)
#plt.savefig('critical_lin.pdf')
   
from mpl_toolkits.axes_grid.inset_locator import inset_axes

plt.figure(figsize=(20, 10))
ax1 = plt.subplot()
line1, = ax1.plot(np.log10(L_list/2), np.log10(obs[10]), linestyle='--', marker='s', markersize=marker_size, color='red', 
                  linewidth=2.0, label='$r=L/2$')
line11, = ax1.plot(np.log10(L_list/4), np.log10(obs[10]), linestyle='--', marker='d', markersize=marker_size, color='green', 
                  linewidth=2.0, label='$r=L/4$')
plt.xlabel('$\log \, r$', fontsize=label_font)
plt.ylabel('$\log \, G(r) $', fontsize=label_font)
plt.legend(loc='upper right', fontsize=legend_font)
ax1.locator_params(axis='x', nbins=5)
ax1.locator_params(axis='y', nbins=2)

inset_axes = inset_axes(ax1, 
                    width="30%", # width = 30% of parent_bbox
                    height="50%", # height : 1 inch
                    loc='lower left')

plt.plot(np.log10(L_list), np.log10(obs[2]), linestyle='--', marker='o', markersize=marker_size, color='blue', linewidth=2.0)
plt.xlabel('$\log \, L$', fontsize=label_font - 8)
inset_axes.xaxis.set_label_coords(0.7,0.2)
inset_axes.yaxis.set_label_coords(0.14,0.4)
plt.ylabel('$\log \, \chi $', fontsize=label_font - 8)
plt.xticks([])
plt.yticks([])

ax1.text(0.57, -0.28, 'a', horizontalalignment='center', verticalalignment='center', 
                 fontweight='bold', fontsize=text_size)
inset_axes.text(1.25, 2.45, 'b', horizontalalignment='center', verticalalignment='center', 
                 fontweight='bold', fontsize=text_size)


plt.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.16)
plt.savefig('critical_lin.pdf')