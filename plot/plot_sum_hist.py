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
from matplotlib import rcParams
rcParams.update({'font.size': 32})

NAME = 'Simple2D16relu_L3_64_16_16_K3333_MReg0.10EReg0.30B1000_OLD'

def read_data(iT):
    ## Read MC and SR data ##
    mc = df.temp_partition(df.read_file(df.data_directory_select(), L=16), iT)
    output = np.load('%s/%s/T%.4f.npy'%(output_dir, NAME, T_list[iT]))
    sampled = (output > np.random.random(output.shape)).astype(np.int)
    
    return block_sum(mc), block_sum(output), block_sum(np.round(output)), block_sum(sampled)

def plot_block_sum(iT=[10,15,20], figsize=(10, 6), save=False):
    ## Plots three lines of four sum histograms ##    
    text = ['a)', 'b)', 'c)']
    
    plt.figure(figsize=figsize)
    for line in range(3):
        mcs, cont, rnd, samp = read_data(iT[line])
        
        plt.subplot(3, 4, 1 + 4*line)
        plt.hist(mcs.ravel(), bins=5)
        plt.ylim((0,350000))
        plt.xticks(np.arange(0, 5, step=1))
        plt.yticks([])
        plt.text(-1, 300000, text[line], horizontalalignment='center', 
                 verticalalignment='center', fontsize=50)
        if line == 0:
            plt.title('MC')
        
        plt.subplot(3, 4, 2 + 4*line)
        plt.hist(cont.ravel(), bins=5)
        plt.ylim((0,350000))
        plt.xticks(np.arange(0, 5, step=1))
        plt.yticks([])
        if line == 0:
            plt.title('Continuous')
        
        plt.subplot(3, 4, 3 + 4*line)
        plt.hist(rnd.ravel(), bins=5)
        plt.ylim((0,350000))
        plt.xticks(np.arange(0, 5, step=1))
        plt.yticks([])
        if line == 0:
            plt.title('Rounded')
        
        plt.subplot(3, 4, 4 + 4*line)
        plt.hist(samp.ravel(), bins=5)
        plt.ylim((0,350000))
        plt.xticks(np.arange(0, 5, step=1))
        plt.yticks([])
        if line == 0:
            plt.title('Sampled')
    
    name = 'sum_hist' + NAME
    for i in iT:
        name += '_T%.4f'%T_list[i]
    name += '.pdf'
    if save:
        plt.savefig(name)
    else:
        plt.show()
        
        