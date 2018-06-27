# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 12:00:24 2018

@author: Stavros
"""

import tensorflow as tf
from os import path, mkdir, listdir
from keras.backend.tensorflow_backend import set_session

def set_GPU_memory(fraction=0.3):    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    set_session(tf.Session(config=config))
    
def create_directory(d):
    ## Create directory if it doesn't exist ##
    if not path.exists(d):
        mkdir(d)
        
def list_networks(network_dir):
    return listdir(network_dir)

def reg_from_name(name):
    i = 4
    while name[i:i+4] != 'MReg':
        i += 1
    i += 4
    mag_reg = float(name[i:i+4])
    i += 8
    en_reg = float(name[i:i+4])
    
    if mag_reg == 0.0 and en_reg == 0.0:
        return False
    else:
        return True

def load_model_from_list(network_dir, list_ind):
    from keras.losses import mean_squared_error
    from keras.models import load_model
    
    network = listdir(network_dir)[list_ind]
    
    metrics_list = ['round_loss', 'cont_loss']
    if reg_from_name(network):
        metrics_list.append('regularization')
        
    custom_objects = {'loss': mean_squared_error}
    for x in metrics_list:
        custom_objects[x] = mean_squared_error
    
    return (network, 
            load_model('%s/%s'%(network_dir, network), custom_objects=custom_objects))