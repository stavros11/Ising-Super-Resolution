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
from keras.losses import mean_squared_error
#from keras.models import Model
#from keras.layers import Lambda, concatenate

######################################################
############ Calculation functions on keras ##########
############ to use for loss                ##########
######################################################

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
        
    return -(Ex + Ey)[:, 0] / n_spins

def block_sumK(x, L):
    state = x[:, :, :, 0]
    ### Calculate sums (part of the block RG function)
    ### Useful to test some heuristics
    sumx = state[:, :, 1:] + state[:, :, :-1]
    # Truncate by keeping only the even terms that belong to the same block
    sumx_trunc = K.transpose(K.gather(K.transpose(sumx), K.arange(0,L,2)))       
    #Same for the y direction by starting from the truncated state
    sumy = sumx_trunc[:, 1:, :] + sumx_trunc[:,:-1,:]
    sumy = K.gather(K.permute_dimensions(sumy, (1, 0, 2)), K.arange(0,L,2))
    return K.permute_dimensions(sumy, (1, 0, 2))


######################################
############ LOSS FUNCTIONS ##########
######################################
    
def cross_entropy_loss(y_true, y_pred, eps=0.0000001):
    cross_entropy = y_true * K.log(y_pred + eps) + (1.0 - 
                                  y_true) * K.log(1.0 - y_pred + eps)
    return - K.mean(cross_entropy, axis=(1,2,3))
    

def round_loss(y_true, y_pred):
    return K.mean(mean_squared_error(K.round(y_true), K.round(y_pred)), 
                  axis=(1, 2))
    
def cont_loss(y_true, y_pred):
    return K.mean(mean_squared_error(y_true, y_pred), axis=(1, 2))

def regularization(y_true, y_pred, mag_reg, en_reg):
    y_true = 2 * y_true - 1
    y_pred = 2 * y_pred - 1
    
    mag_dif = K.square(calculate_magnetization(y_true) - 
                       calculate_magnetization(y_pred))
    en_dif  = K.square(calculate_energy2D(y_true) - calculate_energy2D(y_pred))
    
    return mag_reg * mag_dif + en_reg * en_dif

def create_loss(y_true, y_pred, ce=True):
    if ce:
        return cross_entropy_loss(y_true, y_pred)
    else:
        return K.mean(mean_squared_error(y_true, y_pred), axis=(1,2))
    
def create_loss_reg(y_true, y_pred, ce=True, mag_reg=0.0, en_reg=0.0):
    loss = create_loss(y_true, y_pred, ce=ce)
    return loss + regularization(y_true, y_pred, mag_reg=mag_reg, en_reg=en_reg)
    

######################################################
############ Not ready yet                ############
######################################################

## The idea here is to apply the whole quantity calculation
## on graph (many problems yet!)

#def calculation_model(model, quantities=['magnetization', 'energy']):
#    ## Returns a model with two outputs
#    ## One is the standard CNN output and the other is the quantities output
#    
#    # Dictionary that matches quantities to functions
#    quant_dict = {'magnetization': calculate_magnetization, 
#                  'energy': calculate_energy2D}
#    
#    pred_out = model.output
#
#    sampled = Lambda(lambda x: K.cast(
#            K.greater(x, K.random_uniform(model.output_shape)), 
#            'float16'))(pred_out)
#    calc = Lambda(lambda x: 2*x-1)(sampled)
#    
#    # Create layers
#    l = []
#    for q in quantities:
#        l.append(Lambda(quant_dict[q])(calc))
#    calc = concatenate(l)
#    
#    return Model(model.input, calc)
    
    
    