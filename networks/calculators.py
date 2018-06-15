# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:01:41 2018

@author: Stavros
"""

###########################################################################
############ Functions that calculate thermodynamic quantities ############
############ on keras graph                                    ############
###########################################################################

import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, concatenate

def calculate_magnetization(state):
    return K.expand_dims(K.mean(state, axis=(1, 2, 3)))
    
def calculate_energy2D(state):
    n_spins = int(state.shape[1])**2
    # Energy from x interactions
    Ex = K.sum(state[:,:,1:] * state[:,:,:-1], axis=(1, 2))
    # Energy from y interactions
    Ey = K.sum(state[:,1:] * state[:,:-1], axis=(1, 2))
        
    # Fix periodic boundary conditions
    Ex+= K.sum(state[:,:,0] * state[:,:,-1], axis=1)
    Ey+= K.sum(state[:,0,:] * state[:,-1,:], axis=1)
        
    return K.expand_dims(-(Ex + Ey)[:, 0] / n_spins)


def calculation_model(model, quantities=['magnetization', 'energy']):
    ## Returns a model with two outputs
    ## One is the standard CNN output and the other is the quantities output
    
    # Dictionary that matches quantities to functions
    quant_dict = {'magnetization': calculate_magnetization, 
                  'energy': calculate_energy2D}
    
    pred_out = model.output

    sampled = Lambda(lambda x: K.cast(
            K.greater(x, K.random_uniform(model.output_shape)), 
            'float16'))(pred_out)
    calc = Lambda(lambda x: 2*x-1)(sampled)
    
    # Create layers
    l = []
    for q in quantities:
        l.append(Lambda(quant_dict[q])(calc))
    calc = concatenate(l)
    
    return Model(model.input, calc)
    
    
    