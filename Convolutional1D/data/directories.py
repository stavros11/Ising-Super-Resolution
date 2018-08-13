# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 15:51:06 2018

@author: Stavros
"""

######################################################
############ File directories definitions ############
######################################################

## Temperature list for mc data ##
from numpy import linspace
T_list = linspace(0.01, 3.515, 32)

BASIC_DIR = '/home/sefthymiou/super-resolving-ising/'

## Data directories ##
mc_train_dir = BASIC_DIR + 'ising-data/ising1D-data/ising-1d-N%d-samples%d-train.npy'
mc_test_dir = BASIC_DIR + 'ising-data/ising1D-data/ising-1d-N%d-samples%d-test.npy'

## Network directories ##
models_save_dir = BASIC_DIR + 'Models1D'
metrics_save_dir = BASIC_DIR + 'Metrics1D'

## Quantities directories ##
quantities_dir = BASIC_DIR + 'Quantities1D'
quantities_real_dir = BASIC_DIR + 'QuantitiesReal1D'
quantities_rep_dir = BASIC_DIR + 'QuantitiesRep1D'

## Output directories ##
output_dir = BASIC_DIR + 'Output1D'