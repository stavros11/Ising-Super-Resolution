# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:32:26 2018

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import utils.data_functions as df
from plot_directories import T_list
from plot_directories import output_dir
from utils.decimations import block_sum
from os import path

def read_sr(name):
    sr = np.load(path.join(output_dir, name, 'T%.4f.npy'%T_list[iT]))
    if len(sr.shape) > 3:
        sr = sr[:, :, :, 0]
        
    return sr

def calc_nn(states):
    x = 2 * states - 1
    
    nn1, nn2 = np.zeros(x.shape), np.zeros(x.shape)
    
    nn1[:, :-1] = x[:, 1:] * x[:, :-1]
    nn1[:, -1] = x[:, 0] * x[:, -1]
    
    nn2[:, :, :-1] = x[:, :, 1:] * x[:, :, :-1]
    nn2[:, :, -1] = x[:, :, 0] * x[:, :, -1]
    
    return np.array([nn1, nn2]).transpose((1, 2, 3, 0))

iT = 30
mc = df.temp_partition(df.read_file(df.data_directory_select(1), L=16), iT)
sr = [#read_sr('Simple2D16relu_L2_64_32_K513_PBC_MReg0.00EReg0.00B1000'),
#      read_sr('FeatExt1L2D16relu_L2_32_32_K353_PBC_MReg0.00EReg0.50B1000'),
#      read_sr('Simple2D16relu_L3_64_16_16_K3333_MReg0.10EReg0.30B1000'),
      read_sr('Simple2D16relu_L3_64_16_16_K3333_MReg0.10EReg0.30B1000_OLD')]

sr_s = [(x > np.random.random(x.shape)).astype(np.int) for x in sr]

nn_mc = calc_nn(mc)
nn_s, nn_round = [calc_nn(x) for x in sr_s], [calc_nn(np.round(x)) for x in sr]

bs_mc = block_sum(mc)
bs_s, bs_round = [block_sum(x) for x in sr_s], [block_sum(np.round(x)) for x in sr]

def plot_histograms(figsize=(10,4)):
    n_figs = len(bs_s)
    
    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.hist(bs_mc.ravel(), bins=5)
    plt.ylim((0,350000))
    plt.subplot(132)
    plt.hist(block_sum(sr_s).ravel(), bins=5)
    plt.ylim((0,350000))
    plt.subplot(133)
    plt.hist(block_sum(sr_old_s).ravel(), bins=5)
    plt.ylim((0,350000))
    
    plt.show()
    
def plot_confs_cont(figsize=(10,4)):
    ind = np.random.randint(0, len(mc))
    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.imshow(mc[ind], cmap='Greys', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(132)
    plt.imshow(sr[ind], cmap='Greys', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(133)
    plt.imshow(sr_old[ind], cmap='Greys', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.show()
