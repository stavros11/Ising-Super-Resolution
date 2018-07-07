# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 19:04:04 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
import utils.data_functions as df
from plot_directories import T_list, output_dir
from utils.decimations import block_sum, block_rg_WD
from matplotlib import rcParams
rcParams.update({'font.size': 32})

NAME = 'Simple2D16relu_L2_64_32_K777_PBC_MReg0.00EReg0.30B1000'
iT = 20

## Read MC and SR data ##
mc = df.temp_partition(df.read_file(df.data_directory_select(1), L=16), iT)
output = np.load('%s/%s/T%.4f.npy'%(output_dir, NAME, T_list[iT]))[:,:,:,0]
sampled = (output > np.random.random(output.shape)).astype(np.int)

def plot_block_sum(figsize=(10, 6), save=False):
    ## Plots four sum histograms ##
    mcs = block_sum(mc)
    cont = block_sum(output)
    rnd = block_sum(np.round(output))
    samp = block_sum(sampled)    
    
    plt.figure(figsize=figsize)
    #plt.suptitle('T = %.4f'%T_list[iT])
    
    plt.subplot(141)
    plt.hist(mcs.ravel(), bins=5)
    plt.ylim((0,350000))
    plt.xticks(np.arange(0, 5, step=1))
    plt.yticks([])
    plt.title('MC')
    
    plt.subplot(142)
    plt.hist(cont.ravel(), bins=5)
    plt.ylim((0,350000))
    plt.xticks(np.arange(0, 5, step=1))
    plt.yticks([])
    plt.title('Continuous')
    
    plt.subplot(143)
    plt.hist(rnd.ravel(), bins=5)
    plt.ylim((0,350000))
    plt.xticks(np.arange(0, 5, step=1))
    plt.yticks([])
    plt.title('Rounded')
    
    plt.subplot(144)
    plt.hist(samp.ravel(), bins=5)
    plt.ylim((0,350000))
    plt.xticks(np.arange(0, 5, step=1))
    plt.yticks([])
    plt.title('Sampled')
    
    if save:
        plt.savefig('sum_hist_%s_T%.4f.pdf'%(NAME, T_list[iT]))
    else:
        plt.show()
        
def output_hist(figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.title('T = %.4f'%T_list[iT])
    plt.hist(output.ravel(), bins=20)
    plt.show()

def plot_configs(sampling=0, i_sample=None, figsize=(12, 6), save=False):
    ## Plots MC, RG and SR configurations ##
    ## sampling: 0=Continuous, 1=Sampled, 2=Rounded
    ## i_sample: Which configuration to print (leave None for random)
    
    if i_sample == None:
        i_sample = np.random.randint(0, len(mc))
    small = block_rg_WD(mc)
        
    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.imshow(mc[i_sample], cmap='Greys', vmin=0, vmax=1)
    plt.subplot(132)
    plt.imshow(small[i_sample], cmap='Greys', vmin=0, vmax=1)
    plt.subplot(133)
    if sampling == 0:
        plt.imshow(output[i_sample], cmap='Greys', vmin=0, vmax=1)
    elif sampling == 1:
        plt.imshow(sampled[i_sample], cmap='Greys', vmin=0, vmax=1)
    elif sampling == 2:
        plt.imshow(np.round(output[iT, i_sample]), cmap='Greys', vmin=0, vmax=1)
    else:
        print('Invalid Sampling')
    
    if save:
        plt.savefig('confs%.4f.pdf'%T_list[iT])
    else:
        plt.show()
    
    print(i_sample)