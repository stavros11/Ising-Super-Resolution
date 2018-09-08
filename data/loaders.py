# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 15:57:12 2018

@author: Stavros
"""

import numpy as np
from os import path, mkdir
from directories import mc_train_dir, mc_test_dir
from directories import mc_critical_train_dir, mc_critical_test_dir
from decimations import block_rg, block_rg_WD


################################################
########## DATA FOR TRAINING READERS  ##########
################################################

class TrainingData():
    def __init__(self, args):
        if args.CR:
            train_out = read_file_critical(L=args.L, n_samples=args.nTR, train=True)
            test_out = read_file_critical(L=args.L, n_samples=args.nTE, train=False)
        else:
            train_out = read_file(L=args.L, n_samples=args.nTR, train=True)
            test_out = read_file(L=args.L, n_samples=args.nTE, train=False)
        
        if args.RGWD:
            train_in = block_rg_WD(train_out)
            test_in = block_rg_WD(test_out)
        else:
            train_in = block_rg(train_out)
            test_in = block_rg(test_out)
        
        self.train_in, self.train_out = (add_index(train_in), 
                                         add_index(train_out))
        self.test_in, self.test_out = (add_index(test_in), 
                                       add_index(test_out))
        
class TestData():
    def __init__(self, args):
        if args.CR:
            test_out = read_file_critical(L=args.L, n_samples=args.nTE, train=False)
        else:
            test_out = read_file(L=args.L, n_samples=args.nTE, train=False)
        
        if args.RGWD:
            test_in = block_rg_WD(test_out)
        else:
            test_in = block_rg(test_out)

        self.test_in, self.test_out = (add_index(test_in), 
                                       add_index(test_out))

        
###################################
########## LOAD MC DATA  ##########
###################################

def read_file(L=16, q=2, n_samples=10000, train=False):
    ## Returns dataset normilized to [0,1] ##
    if train:
        data = np.load(mc_train_dir%(n_samples, L, q))
    else:
        data = np.load(mc_test_dir%(n_samples, L, q))
    return data.reshape(data.shape[0], L, L)

def read_file_critical(L=16, n_samples=40000, train=False):
    if train:
        return np.load(mc_critical_train_dir%(n_samples, L))
    else:
        return np.load(mc_critical_test_dir%(n_samples, L))
    
def read_file_GPU(L=16, q=2):
    data = np.load('/home/data/critical-2d-ising/L=%d/q=%d/configs.npy'%(L, q))
    return data.reshape(data.shape[0], L, L)    

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