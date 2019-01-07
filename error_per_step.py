# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 16:38:08 2019

@author: Stavros
"""

import numpy as np
import h5py
from os import path
from keras.models import load_model
from keras.losses import mean_squared_error
from data.directories import models_save_dir, T_list
from networks.architectures import duplicate_simple2D
from networks.ising import get_observables_with_corr_and_tpf

custom_list = ['loss', 'cont_loss', 'round_loss', 'regularization']
Tc = 2.0 / np.log(1 + np.sqrt(2))
ind_renorm = [10, 12, 13, 14]
n_upsamplings = 3

### Load basic model ###
custom_objects = {}
for x in custom_list:
    custom_objects[x] = mean_squared_error
    
def load_model_at(T): 
    return load_model(path.join(
            models_save_dir, 'Simple2D16relu_L3_64_16_16_K3333_MReg0.10EReg0.30_OLD', 
        'T%.4f.h5'%T), custom_objects=custom_objects)

### Load basic data ###
def load_data(dr):
    h5f = h5py.File(dr, 'r')
    data = []
    for x in h5f.keys():
        data.append(h5f[x][:])
    h5f.close()
    return np.concatenate(tuple(data), axis=0)

mydir = "C:/Users/Stavros.SAVVAS-PROBOOK/Documents/Ising_MC/"
confs_basic = load_data(path.join(mydir, 'confs/critical_half/L=16', 'data_L16.h5'))
confs = np.copy(confs_basic)[:, :, :, np.newaxis]
obs = [get_observables_with_corr_and_tpf(confs_basic, Tc/2.0)]

for iUP in range(n_upsamplings):
    T = T_list[ind_renorm[iUP]]
    basic_model = load_model_at(T)
    model = duplicate_simple2D(basic_model, confs.shape, hid_filters=[64, 16, 16],
                               kernels=[3, 3, 3, 3])
    
    sr_cont = model.predict(confs_basic)
    confs = (sr_cont > np.random.random(sr_cont.shape))
    obs.append(get_observables_with_corr_and_tpf(confs[:, :, :, 0], T))
    
    print('%d / %d upsamplings done'%(iUP+1, n_upsamplings))