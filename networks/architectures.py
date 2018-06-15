# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:50:55 2018

@author: Stavros
"""

##################################################################
############ Functions that create basic keras models ############
############ written using the Sequential model       ############
##################################################################

from keras.models import Sequential
from keras.layers import Conv2D, InputLayer, UpSampling2D, Lambda

###################################
############ UTILITIES ############
###################################

def PBCLayer2D(x, pad=1):
    from keras.backend import tile
    L = int(x.shape[1])
    y = tile(x, [1, 2, 2, 1])
    
    return y[:, :L+pad, :L+pad]

###################################
########## ARCHITECTURES ##########
###################################

## x: shape of input (samples, x, y, channels)
## hid_filters: list with filters for hidden layers
## hid_kernels: list with kernel size (square kernel)
## kernels must have one more entry than filters for last layer
## for which the filter is by default 1
    
## Last activation is by default sigmoid to interpret the output as probability!

def simple2D(x, hid_filters=[64, 32], kernels=[6, 1, 3], hid_act='relu'):
    model = Sequential()
    model.add(InputLayer(input_shape=x[1:]))
    model.add(UpSampling2D())
    for (k, f) in zip(kernels, hid_filters):
        model.add(Conv2D(f, k, padding='same', 
                         kernel_initializer='he_normal', activation=hid_act))
    model.add(Conv2D(1, kernels[-1], padding='same', 
                     kernel_initializer='glorot_normal', activation='sigmoid'))
    
    return model

def simple2D_pbc(x, hid_filters=[64, 32], kernels=[6, 1, 3], hid_act='relu'):
    model = Sequential()
    model.add(InputLayer(input_shape=x[1:]))
    model.add(UpSampling2D())
    for (k, f) in zip(kernels, hid_filters):
        model.add(Lambda(PBCLayer2D, arguments={'pad' : k-1}))
        model.add(Conv2D(f, k, padding='valid', 
                         kernel_initializer='he_normal', activation=hid_act))

    model.add(Lambda(PBCLayer2D, arguments={'pad' : kernels[-1]-1}))
    model.add(Conv2D(1, kernels[-1], padding='valid', 
                     kernel_initializer='glorot_normal', activation='sigmoid'))
    
    return model

###################################
########### DUPLICATES  ###########
###################################

## They read the weights from a model ##
def duplicate_simple2D(old_model, x, hid_filters=[64, 32], 
                     kernels=[6, 1, 3], hid_act='relu'):
    # x: new input dimension
    new_model = simple2D(x, hid_filters=hid_filters, hid_act=hid_act, 
                         kernels=kernels)
    for i in range(1, len(old_model.layers)):
        new_model.layers[i].set_weights(
                old_model.layers[i].get_weights())

    return new_model

def duplicate_simple2D_pbc(old_model, x, hid_filters=[64, 32], 
                           kernels=[6, 1, 3], hid_act='relu'):
    # x: new input dimension
    new_model = simple2D_pbc(x, hid_filters=hid_filters, 
                                      hid_act=hid_act, kernels=kernels)
    for i in range(1, len(old_model.layers)):
        new_model.layers[i].set_weights(
                old_model.layers[i].get_weights())

    return new_model

##################################################
########## FUNCTIONS THAT RETURN NAMES  ##########
##################################################
    
def get_name(x, hid_filters=[64, 32], kernels=[6, 1, 3], hid_act='relu',
             pbc=False):
    ## Returns name (that contains model info)
    
    # Create name and write filters
    name = 'Simple2D%d%s_L%d'%(x[1], hid_act, len(hid_filters))
    for i in hid_filters:
        name += '_' + str(i)
        
    # Add kernel info
    name += '_K'
    for i in kernels:
        name += str(i)
        
    if pbc:
        name += '_PBC'
        return name
    else:
        return name
    
def get_model(x, hid_filters=[64, 32], kernels=[6, 1, 3], hid_act='relu',
              pbc=False):
    ## Returns model
    if pbc:
        return simple2D_pbc(x, hid_filters=hid_filters, kernels=kernels, 
                            hid_act=hid_act)
    else:
        return simple2D(x, hid_filters=hid_filters, kernels=kernels, 
                        hid_act=hid_act)

def get_name_and_model(x, hid_filters=[64, 32], kernels=[6, 1, 3], hid_act='relu',
                       pbc=False):
    ## Returns name (that contains model info), model
    
    name = get_name(x, hid_filters=hid_filters, kernels=kernels, hid_act=hid_act,
                    pbc=pbc)
    
    if pbc:
        return name, simple2D_pbc(x, hid_filters=hid_filters, kernels=kernels, 
                                  hid_act=hid_act)
    else:
        return name, simple2D(x, hid_filters=hid_filters, kernels=kernels, 
                              hid_act=hid_act)
    