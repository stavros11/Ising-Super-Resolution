# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 19:30:31 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
import plot.utils.data_functions_critical as df
from plot.plot_directories import models_critical_dir
from networks.architectures import simple2D_pbc, duplicate_simple2D_pbc

#64652 i_sample
NAME = 'Simple2D16relu_L2_64_32_K333_PBC_MReg0.00EReg0.00B1000_Ver2Run7'

mc = df.read_file(df.data_directory_select(1), n_samples=100000)

model0 = simple2D_pbc([1, 16, 16, 1], kernels=[3,3,3])
model0.load_weights('%s/%s.hdf5'%(models_critical_dir, NAME))


def plot_mc_only(i_sample=None, figsize=(5, 5)):
    if i_sample == None:
        i_sample = np.random.randint(0, len(mc))
    
    print(i_sample)
    plt.figure(figsize=figsize)
    plt.imshow(mc[i_sample], cmap='Greys', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def upsample(i_sample, model, upsamplings):
    states = [mc[i_sample]]
    
    for i in range(upsamplings):
        model = duplicate_simple2D_pbc(model, (1,)+states[-1].shape+(1,), 
                                       hid_filters=[64, 32], kernels=[3, 3, 3])
        
        cont = model.predict(states[-1].reshape(
                (1,) + states[-1].shape + (1,)))[0,:,:,0]
        
        states.append((cont > np.random.random(cont.shape)).astype(np.int))
    
    return states

def plot_sr(model, i_sample=None, figsize=(15, 6), p_list=[0, 1], save=False):
    if i_sample == None:
        i_sample = np.random.randint(0, len(mc))
        
    states = upsample(i_sample, model, p_list[-1]+1)
    n = len(p_list)
    
    plt.figure(figsize=figsize)
    for (i, p) in enumerate(p_list):
        plt.subplot(100 + 10 * n + i + 1)
        plt.title('%d x %d'%(2**(4+p), 2**(4+p)))
        plt.imshow(states[p], cmap='Greys', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
    
    if save:
        plt.savefig('Confs_%s.pdf'%NAME)
    else:
        plt.show()
        
def plot_sr_states(states, figsize=(15, 6), p_list=[0, 1], save=False):
    n = len(p_list)
    
    plt.figure(figsize=figsize)
    for (i, p) in enumerate(p_list):
        plt.subplot(100 + 10 * n + i + 1)
        plt.title('%d x %d'%(2**(4+p), 2**(4+p)))
        plt.imshow(states[p], cmap='Greys', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
    
    if save:
        plt.savefig('Confs_%s.pdf'%NAME)
    else:
        plt.show()
    