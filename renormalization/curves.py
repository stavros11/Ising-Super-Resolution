# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:28:41 2018

@author: Stavros
"""

import numpy as np
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def curve(x, a, b):
    return a / np.log(np.cosh(b / x))

def inv_curve(x, a, b):
    return b / np.arccosh(np.exp(a / x))

obs = np.load('MCvsRG_Observables_L16.npy')
aM, bM = np.load('Magnetization_Transformation_Params_L16.npy')
aE, bE = np.load('Energy_Transformation_Params_L16.npy')
T_list = np.linspace(0.01, 3.515, 32)

fig = plt.figure(figsize=(30, 14))
matplotlib.rcParams.update({'font.size': 42})
label_font = 16
text_font = 58

# set height ratios for sublots
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1]) 

# the fisrt subplot
ax0 = plt.subplot(gs[0])
line_mcM, = ax0.plot(T_list, obs[0, :, 0], color='blue', linewidth=3.5, alpha=1.0)
line_rgM, = ax0.plot(T_list, obs[1, :, 0], linestyle='--', color='black', 
                     alpha=0.5, linewidth=3.5, marker='s', markersize=12)
line_srM, = ax0.plot(T_list, obs[2, :, 0], color='green', linewidth=3.5, 
                     alpha=1.0)
line_FM, = ax0.plot(curve(T_list, aM, bM), obs[1, :, 0], color='red',
                    alpha=1.0, linewidth=3.5, marker='d', markersize=12)
plt.ylabel('$M$', fontsize=label_font)

#the second subplot
ax1 = plt.subplot(gs[2], sharex = ax0)
line_mcE, = ax1.plot(T_list, obs[0, :, 1], color='blue', linewidth=4.5)
line_rgE, = ax1.plot(T_list, obs[1, :, 1], color='blue', linewidth=4.5, alpha=0.4)
line_srE, = ax1.plot(T_list, obs[2, :, 1], 'o--', color='red', linewidth=3.0, markersize=12)
plt.setp(ax0.get_xticklabels(), visible=False)
plt.ylabel('$E$', fontsize=label_font)

ax0.locator_params(axis='y', nbins=5)
ax1.locator_params(axis='y', nbins=5)
# remove last tick label for the second subplot
yticks = ax0.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)

# put lened on first subplot
ax1.legend((line_mcM, line_srM, line_rgM), 
           (''.join(['16', r'$\times$', '16', r' MC']),
            ''.join([r'8', r'$\times$', r'8', r' MC']),
           ''.join([r'8', r'$\times$', r'8', r' RG'])),
           loc='upper left', fontsize=52)

plt.text(0.15, 0.85, 'a', horizontalalignment='center', verticalalignment='center', 
                 fontweight='bold', fontsize=text_font)
plt.text(0.15, -0.15, 'b', horizontalalignment='center', verticalalignment='center', 
                 fontweight='bold', fontsize=text_font)

plt.xlabel('$T$', fontsize=label_font)
plt.xlim([0, 3.7])
plt.locator_params(axis='x', nbins=8)
# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)


plt.savefig('MCvsRG.pdf')
