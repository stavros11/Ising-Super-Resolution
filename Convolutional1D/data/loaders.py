# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 15:57:12 2018

@author: Stavros
"""

import numpy as np
from os import path, mkdir
from directories import mc_train_dir, mc_test_dir

################################################
########## DATA FOR TRAINING READERS  ##########
################################################

class TrainingData():
    def __init__(self, args):
        train_out = read_file(L=args.L, n_samples=args.nTR, train=True)
        test_out = read_file(L=args.L, n_samples=args.nTE, train=False)
        
        train_in = block_rg(train_out)
        test_in = block_rg(test_out)
        
        self.train_in, self.train_out = (add_index(train_in), 
                                         add_index(train_out))
        self.test_in, self.test_out = (add_index(test_in), 
                                       add_index(test_out))
        
class TestData():
    def __init__(self, args):
        test_out = read_file(L=args.L, n_samples=args.nTE, train=False)
        test_in = block_rg(test_out)

        self.test_in, self.test_out = (add_index(test_in), 
                                       add_index(test_out))

##################################
########## DECIMATIONS  ##########
##################################

def block_rg(state):
    (n_samples, N) = state.shape
    return state[:, np.arange(0, N, 2)]
    
###################################
########## LOAD MC DATA  ##########
###################################

def read_file(L=32, n_samples=10000, train=False):
    ## Returns dataset normilized to [0,1] ##
    if train:
        data = np.load(mc_train_dir%(L, n_samples))
    else:
        data = np.load(mc_test_dir%(L, n_samples))
    return data.reshape((data.shape[0], L))
    
####################################
########## DATA UTILITIES ##########
####################################

def add_index(data):
    ## Adds a 1-component dimension to a numpy array to use as input to CNN ##
    return data.reshape(data.shape+(1,))

def temp_partition(data, iT, n_samples=10000):
    return data[iT * n_samples : (iT+1) * n_samples]

def create_directory(d):
    ## Create directory if it doesn't exist ##
    if not path.exists(d):
        mkdir(d)