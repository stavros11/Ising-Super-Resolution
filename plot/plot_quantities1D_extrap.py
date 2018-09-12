# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 09:28:18 2018

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
from plot_directories import quantities_dir1D_fixed
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
NAME = 'Simple1D32relu_L1_32_K53_PBC_MReg0.00EReg0.00B1000_extr'
obs = np.load('%s/%s.npy'%(quantities_dir1D_fixed, NAME))

# Use rounding instead of sampling for the five lowest temperatures 
# to correct noise in susc and Cv
obs[:5, -1] = obs[:5, -2] 
    
matplotlib.rcParams.update({'font.size': 38})
test_size = 50 #text
label_size = 46

from mpl_toolkits.axes_grid.inset_locator import inset_axes

T_ren = 2.0 / np.arccosh(np.exp(2.0 / T_list))
def get_errors():
    errors = np.zeros([len(T_list), obs.shape[-1]])
    for (iT, T) in enumerate(T_ren):
        dif = np.abs(T - T_list)
        i1 = dif.argmin()
        dif[i1] = 1000
        i2 = dif.argmin()
        mc_values = (obs[i2, 0] - obs[i1, 0]) * (T - T_list[i1])/ (T_list[i2] - T_list[i1])
        mc_values+= obs[i1, 0]
        
        errors[iT] = np.abs((mc_values - obs[iT, -1]) / mc_values)
        
    return errors
        
errors = get_errors()

fig = plt.figure(figsize=(30, 7))
#gs = gridspec.GridSpec(1, 2)
#cmap = plt.cm.get_cmap('Paired_r', lut=4)
cp = sns.color_palette("Paired", 10)

ax2 = fig.add_subplot(121)
line_mcMf, = plt.plot(T_list, obs[:, 0, 0], color=cp[1], alpha=0.8, linewidth=3.5)
line_rgMf, = plt.plot(T_list, obs[:, 1, 0], color=cp[0], linewidth=3.5, linestyle='--')
line_srMf, = plt.plot(T_ren, obs[:, -1, 0], color=cp[4], linestyle='', markersize=12, marker='o')
plt.ylabel('$M$', fontsize=label_size)
plt.xlabel('$T$', fontsize=label_size)

plt.text(0, 0.97, 'a', horizontalalignment='center', verticalalignment='center', 
         fontweight='bold', fontsize=test_size)

ax_ins = inset_axes(ax2, 
                    width="40%", # width = 30% of parent_bbox
                    height="40%", # height : 1 inch
                    loc=1)
plt.plot(T_list, obs[:, 0, 2], color=cp[3], alpha=0.8, linewidth=3)
plt.plot(T_list, obs[:, 1, 2], color=cp[2], linewidth=3, linestyle='--')
plt.plot(T_ren, obs[:, -1, 2], linestyle=' ', 
                     color=cp[-1], alpha=0.8, marker='o', markersize=10)
plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle=(0, (5, 1)), color='k', linewidth=1.5)
plt.locator_params(axis='x', nbins=2)
plt.locator_params(axis='y', nbins=3)
plt.xlim([0, 1.8])
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.ylabel('$\chi $', fontsize=label_size - 10)
plt.xlabel('$T$', fontsize=label_size - 10)


ax3 = fig.add_subplot(122)
line_mcEf, = ax3.plot(T_list, obs[:, 0, 1], color=cp[1], alpha=0.8, linewidth=3.5)
line_rgEf, = ax3.plot(T_list, obs[:, 1, 1], color=cp[0], linewidth=3.5, linestyle='--')
line_srEf, = ax3.plot(T_ren, obs[:, -1, 1], color=cp[4], linestyle='', markersize=12, marker='o')
plt.ylabel('$E$', fontsize=label_size)
plt.xlabel('$T$', fontsize=label_size)

ax3.text(0, -0.32, 'b', horizontalalignment='center', verticalalignment='center', 
         fontweight='bold', fontsize=test_size)

ax_ins2 = inset_axes(ax3, 
                    width="40%", # width = 30% of parent_bbox
                    height="40%", # height : 1 inch
                    loc=4)
plt.plot(T_list, obs[:, 0, 3], color=cp[3], alpha=0.8, linewidth=3)
plt.plot(T_list, obs[:, 1, 3], color=cp[2], linewidth=3, linestyle='--')
plt.plot(T_ren, obs[:, -1, 3], linestyle=' ', 
                     color=cp[-1], alpha=0.8, marker='o', markersize=10)
plt.axvline(x = 2 / np.log(1 + np.sqrt(2)), linestyle=(0, (5, 1)), color='k', linewidth=1.5)
plt.locator_params(axis='x', nbins=2)
plt.locator_params(axis='y', nbins=3)
plt.xlim([0, 1.8])
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.ylabel('$C_V$', fontsize=label_size - 10)
plt.xlabel('$T$', fontsize=label_size - 10)
ax_ins2.xaxis.set_label_position('top') 
ax_ins2.xaxis.tick_top()


ax2.legend((line_mcMf, line_rgMf, line_srMf), ('$N=%d$ MC'%L, '$N=%d$ MC'%(L//2), '$N=%d$ SR'%L), 
           loc='lower left', fontsize=34)

plt.savefig('ups_real1D_extr.pdf', bbox_inches='tight')