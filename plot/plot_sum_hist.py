# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 19:04:04 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
import utils.data_functions as df
from plot_directories import T_list, output_dir
from utils.decimations import block_sum

from seaborn import color_palette
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'STIXGeneral'

NAME = 'Simple2D16relu_L3_64_16_16_K3333_MReg0.10EReg0.30B1000_OLD'

def read_data(iT):
    ## Read MC and SR data ##
    mc = df.temp_partition(df.read_file(df.data_directory_select(1), L=16), iT)
    output = np.load('%s/%s/T%.4f.npy'%(output_dir, NAME, T_list[iT]))
    sampled = (output > np.random.random(output.shape)).astype(np.int)
    
    return block_sum(mc), block_sum(output), block_sum(np.round(output)), block_sum(sampled)

def plot_block_sum(iT=[10,15,20], figsize=(10, 6), save=False):
    ## Plots three lines of four sum histograms ##    
    text = ['(a)', '(b)', '(c)']
    
    label_size = 60
    
    palette = color_palette('colorblind', desat=0.9)
    plt.figure(figsize=figsize)
    for line in range(3):
        mcs, cont, rnd, samp = read_data(iT[line])
        
        plt.subplot(3, 3, 1 + 3*line)
        plt.hist(mcs.ravel(), bins=5, color=palette[0], alpha=0.6)
        plt.ylim((0,350000))
        plt.xticks(np.arange(0, 5, step=2))
        plt.yticks([])
        plt.text(-0.64, 310000, text[line], horizontalalignment='center', 
                 verticalalignment='center', fontsize=64)
        if line == 0:
            plt.title('MC', fontsize=label_size)
        
#        plt.subplot(3, 3, 2 + 3*line)
#        plt.hist(cont.ravel(), bins=5, color=palette[1])
#        plt.ylim((0,350000))
#        plt.xticks(np.arange(0, 5, step=2))
#        plt.yticks([])
#        if line == 0:
#            plt.title('Continuous', fontsize=label_size)
        
        plt.subplot(3, 3, 2 + 3*line)
        plt.hist(rnd.ravel(), bins=5, color=palette[1], alpha=0.6)
        plt.ylim((0,350000))
        plt.xticks(np.arange(0, 5, step=2))
        plt.yticks([])
        if line == 0:
            plt.title('Rounded', fontsize=label_size)
        
        plt.subplot(3, 3, 3 + 3*line)
        plt.hist(samp.ravel(), bins=5, color=palette[2], alpha=0.6)
        plt.ylim((0,350000))
        plt.xticks(np.arange(0, 5, step=2))
        plt.yticks([])
        if line == 0:
            plt.title('Sampled', fontsize=label_size)
    
    name = 'sum_hist' + NAME
    for i in iT:
        name += '_T%.4f'%T_list[i]
    name += '.pdf'
    if save:
        plt.savefig(name, bbox_inches='tight')
    else:
        plt.show()
        
plot_block_sum(figsize=(25,20), save=True)
        
## What happens with ties (for Roger) ##
#def tie_reason(states):
#    n_samples, Ly, Lx = states.shape
#    bs = block_sum(states)
#    n_ties = (bs == 2).sum()
#    keeper = (bs == 2).astype(np.int).repeat(2, axis=1).repeat(2, axis=2)
#    
#    tied_state = keeper * states
#    
#    rows = (tied_state[:, 1:] * tied_state[:, :-1])[:, np.arange(0, Ly, 2)].sum()
#    cols = (tied_state[:, :, 1:] * tied_state[:, :, :-1])[:, :, np.arange(0, Lx, 2)].sum()
#    
#    return np.array([rows, cols, n_ties - rows - cols])
#    
#mc_sums, sr_sums = np.zeros([32, 3]), np.zeros([32, 3])
#
#for iT in range(32):
#    mc = df.temp_partition(df.read_file(df.data_directory_select(1), L=16), iT)
#    output = np.load('%s/%s/T%.4f.npy'%(output_dir, NAME, T_list[iT]))
#    sampled = (output > np.random.random(output.shape)).astype(np.int)
#    
#    mc_sums[iT] = tie_reason(mc)
#    sr_sums[iT] = tie_reason(sampled)
#    
#    print(iT)
#
#
#rcParams.update({'font.size': 48})
#label_size = 52
#from mpl_toolkits.axes_grid.inset_locator import inset_axes
#
#
#cut_iT = 10
#mc_norm = 100*mc_sums[cut_iT:] / mc_sums[cut_iT:].sum(axis=1)[:, np.newaxis]
#sr_norm = 100*sr_sums[cut_iT:] / sr_sums[cut_iT:].sum(axis=1)[:, np.newaxis]
#
#plt.figure(figsize=(30, 8))
#plt.subplot(121)
#plt.plot(T_list[cut_iT:], mc_norm[:, 0], color='blue', label='Row', linewidth=3.5)
#plt.plot(T_list[cut_iT:], mc_norm[:, 1], color='green', linewidth=3.5, label='Column')
#plt.plot(T_list[cut_iT:], sr_norm[:, 0], color='blue', linestyle='--', linewidth=3.5)
#plt.plot(T_list[cut_iT:], sr_norm[:, 1], color='green', linestyle='--', linewidth=3.5)
#plt.ylabel('NN (%)',fontsize=label_size)
#plt.xlabel('$T$', fontsize=label_size)
#plt.legend(fontsize=50, loc='upper right')
#
#ax2 = plt.subplot(122)
#plt.title('$T=2.9313$')
#N, bins, patches = plt.hist([0, 1, 2, 3], bins=4, weights=[mc_sums[20,0]+mc_sums[20,1], mc_sums[20,2], sr_sums[20,0]+sr_sums[20,1], sr_sums[20,2]])
#plt.ylim((0,150000))
#plt.xticks([0, 1, 2, 3], ['MC-NN', 'MC Diag', 'SR-NN', 'SR Diag'], fontsize=44)
#plt.yticks([])
#for i in range(0,2):
#    patches[i].set_facecolor('blue')
#for i in range(2,4):    
#    patches[i].set_facecolor('orange')
#
#ax_ins = inset_axes(ax2, 
#                    width="40%", # width = 30% of parent_bbox
#                    height="40%", # height : 1 inch
#                    loc=1)
#
#plt.plot(T_list[cut_iT:], mc_norm[:, 2], color='red', linewidth=3.5, label='MC')
#plt.plot(T_list[cut_iT:], sr_norm[:, 2], color='red', linestyle='--', linewidth=3.5, label='SR')
#plt.ylabel('Diagonal (%)', fontsize=label_size-10)
#plt.xlabel('$T$', fontsize=label_size-10)
#
#plt.savefig('block_ties_temp.pdf', bbox_inches='tight')