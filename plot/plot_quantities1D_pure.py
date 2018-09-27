# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:56:58 2018

@author: Admin
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 42})

from plot_directories import T_list1D as T_list
from plot_directories import quantities_dir1D
# Use this T_list when plot_directories module is not available
#T_list = np.linspace(0.01, 4.538, 32)


### !!! .NPY DESCRIPTION !!! ###
# obs = (32, 5, 12)
# ind1: temperatures
# ind2: [MC, RG, SR continuous, SR rounded, SR sampled]
# ind3: [Mag, En, Susc, specHeat, Mag2, Mag4, En2, tpf(L/2), 
#        tpf(L/4), S0, S1, S2]

# Load data (fix .npy directory here)
L = 64
NAME = 'Simple1D64relu_L2_32_16_K533_PBC_MReg0.00EReg0.00B1000'
#obs = np.load('%s/%s.npy'%(quantities_dir1D, NAME))
fixed_older = np.load('%s/%s.npy'%(quantities_dir1D, NAME))

# Use rounding instead of sampling for the five lowest temperatures 
# to correct noise in susc and Cv
#obs[:3, -1] = obs[:3, -2]
fixed_older[:5, -1] = fixed_older[:5, -2]
    
    
matplotlib.rcParams.update({'font.size': 38})
test_size = 50 #text
label_size = 46

from mpl_toolkits.axes_grid.inset_locator import inset_axes

T_ren = 2.0 / np.arccosh(np.exp(2.0 / T_list))
def get_errors():
    errors = np.zeros([len(T_list), fixed_older.shape[-1]])
    for (iT, T) in enumerate(T_ren):
        #errors[iT] = np.abs((mc_values - fixed_older[iT, -1]) / mc_values)
        errors[iT] = (fixed_older[iT, -1] - fixed_older[iT, 0])
        
    return errors
        
errors = get_errors()

fig = plt.figure(figsize=(30, 7))
#gs = gridspec.GridSpec(1, 2)
#cmap = plt.cm.get_cmap('Paired_r', lut=4)
cp = sns.color_palette("Paired")

ax2 = fig.add_subplot(121)
line_mcMf, = plt.plot(T_list, fixed_older[:, 0, 0], color=cp[1], alpha=0.8, linewidth=3.5)
line_rgMf, = plt.plot(T_list, fixed_older[:, 1, 0], color=cp[0], linewidth=3.5, linestyle='--')
line_srMf, = plt.plot(T_list, fixed_older[:, -1, 0], color=cp[4], linestyle='', markersize=12, marker='o')
plt.ylabel('$M$', fontsize=label_size)
plt.xlabel('$T$', fontsize=label_size)

plt.text(0, 0.97, 'a', horizontalalignment='center', verticalalignment='center', 
         fontweight='bold', fontsize=test_size)

ax_ins = inset_axes(ax2, 
                    width="40%", # width = 30% of parent_bbox
                    height="40%", # height : 1 inch
                    loc=1)
plt.plot(T_list, 1000* errors[:, 0], linestyle=':', linewidth=2.0, color=cp[3], alpha=0.7,
         markersize=12, marker='^')
#plt.locator_params(axis='x', nbins=5)
#plt.locator_params(axis='y', nbins=3)
plt.xticks(fontsize=32)
plt.yticks([-4, -2, 0, 2], fontsize=32)
plt.yticks(fontsize=32)
plt.ylabel('Error ($10^{-3}$)', fontsize=label_size - 18)
plt.xlabel('$T$', fontsize=label_size - 14)


ax3 = fig.add_subplot(122)
line_mcEf, = ax3.plot(T_list, fixed_older[:, 0, 1], color=cp[1], alpha=0.8, linewidth=3.5)
line_rgEf, = ax3.plot(T_list, fixed_older[:, 1, 1], color=cp[0], linewidth=3.5, linestyle='--')
line_srEf, = ax3.plot(T_list, fixed_older[:, -1, 1], color=cp[4], linestyle='', markersize=12, marker='o')
plt.ylabel('$E$', fontsize=label_size)
plt.xlabel('$T$', fontsize=label_size)

ax2.set_zorder(1)
ax_ins.set_zorder(2)

ax2.text(4.615, 0.96, 'b', horizontalalignment='center', verticalalignment='center', 
         fontweight='bold', fontsize=test_size)

ax_ins2 = inset_axes(ax3, 
                    width="40%", # width = 30% of parent_bbox
                    height="40%", # height : 1 inch
                    loc=4)
plt.plot(T_list, 100 * errors[:, 1], linestyle=':', linewidth=2.0, color=cp[3], alpha=0.7,
         markersize=15, marker='^')
ax_ins2.xaxis.tick_top()
#plt.locator_params(axis='x', nbins=5)
#plt.locator_params(axis='y', nbins=3)
plt.xticks(fontsize=32)
#plt.yticks([-1, 0, 1], fontsize=32)
plt.yticks(fontsize=32)
plt.ylabel('Error ($10^{-2}$)', fontsize=label_size - 18)
plt.xlabel('$T$', fontsize=label_size - 14)
ax_ins2.xaxis.set_label_position('top') 


ax3.legend((line_mcMf, line_rgMf, line_srMf), ('$N=%d$ MC'%L, '$N=%d$ DS'%(L//2), '$N=%d$ SR'%L), 
           loc='upper left', fontsize=34)

plt.savefig('test1D_pure.pdf', bbox_inches='tight')