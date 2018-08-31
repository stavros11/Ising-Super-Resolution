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

iT = 20
mc = df.temp_partition(df.read_file(df.data_directory_select(3), L=16), iT)
sr = read_sr('FeatExt1L2D16relu_L2_32_32_K353_PBC_MReg0.00EReg0.00B1000')
sr_old = read_sr('Simple2D16relu_L3_64_16_16_K3333_MReg0.10EReg0.30B1000_OLD')

sr_s = (sr > np.random.random(sr.shape)).astype(np.int)
sr_old_s = (sr_old > np.random.random(sr_old.shape)).astype(np.int)

nn_list = [calc_nn(x) for x in [mc, sr_s, sr_old_s]]

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.hist(block_sum(mc).ravel(), bins=5)
plt.ylim((0,350000))
plt.subplot(132)
plt.hist(block_sum(sr_s).ravel(), bins=5)
plt.ylim((0,350000))
plt.subplot(133)
plt.hist(block_sum(sr_old_s).ravel(), bins=5)
plt.ylim((0,350000))