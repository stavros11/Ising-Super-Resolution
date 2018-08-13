# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 12:00:24 2018

@author: Stavros
"""

import numpy as np
from os import path, mkdir
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from ising import get_observables

def set_GPU_memory(fraction=0.3):    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    set_session(tf.Session(config=config))
    
def create_directory(d):
    ## Create directory if it doesn't exist ##
    if not path.exists(d):
        mkdir(d)
        
def calculate_observables(data_or, data_in, data_out, T):
    pred_samp = (data_out > np.random.random(data_out.shape)).astype(np.int)
    
    obs = np.zeros([5, 7])
    obs[0] = get_observables(data_or, T)
    obs[1] = get_observables(data_in, T)
    obs[2] = get_observables(data_out, T)
    obs[3] = get_observables(np.round(data_out), T)
    obs[4] = get_observables(pred_samp, T)
    
    return obs

def calculate_observables_real(data_or, data_in, data_out, T, Tr):
    pred_samp = (data_out > np.random.random(data_out.shape)).astype(np.int)
    
    obs = np.zeros([5, 7])
    obs[0] = get_observables(data_or, T)
    obs[1] = get_observables(data_in, T)
    obs[2] = get_observables(data_out, Tr)
    obs[3] = get_observables(np.round(data_out), Tr)
    obs[4] = get_observables(pred_samp, Tr)
    
    return obs

def calculate_observables_rep(data_out, Tr):
    pred_samp = (data_out > np.random.random(data_out.shape)).astype(np.int)
    
    obs = np.zeros([3, 7])
    obs[0] = get_observables(data_out, Tr)
    obs[1] = get_observables(np.round(data_out), Tr)
    obs[2] = get_observables(pred_samp, Tr)
    
    return obs