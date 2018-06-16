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
T_list = linspace(0.01, 4.538, 32)

## Data directories ##
#mc_train_dir = 'C:/Users/Stavros.SAVVAS-PROBOOK/Documents/Scripts_and_Programs/Super_resolution_Titan_scripts/Ising_Data/ising-data-train-%d/L=%d/q=%d/configs.npy'
#mc_test_dir = 'C:/Users/Stavros.SAVVAS-PROBOOK/Documents/Scripts_and_Programs/Super_resolution_Titan_scripts/Ising_Data/ising-data-test-%d/L=%d/q=%d/configs.npy'
#mc_critical_train_dir = 'C:/Users/Stavros.SAVVAS-PROBOOK/Documents/Scripts_and_Programs/Super_resolution_Titan_scripts/Ising_Data/ising-critical-train-%d/L=%d/configs.npy'
#mc_critical_test_dir = 'C:/Users/Stavros.SAVVAS-PROBOOK/Documents/Scripts_and_Programs/Super_resolution_Titan_scripts/Ising_Data/ising-critical-test-%d/L=%d/configs.npy'
mc_train_dir = '~/super-resolving-ising/ising-data-train-%d/L=%d/q=%d/configs.npy'
mc_test_dir = '~/super-resolving-ising/ising-data-train-%d/L=%d/q=%d/configs.npy'
mc_critical_train_dir = '~/super-resolving-ising/ising-data-critical-%d/L=%d/configs.npy'
mc_critical_test_dir = '~/super-resolving-ising/ising-data-critical-%d/L=%d/configs.npy'

## Network directories ##
models_save_dir = '~/super-resolving-ising/Models'
metrics_save_dir = '~/super-resolving-ising/Metrics'

models_critical_save_dir = '~/super-resolving-ising/ModelsCritical'
metrics_critical_save_dir = '~/super-resolving-ising/MetricsCritical'

## Quantities directories ##
quantities_dir = '~/super-resolving-ising/Quantities'
quantities_critical_dir = '~/super-resolving-ising/QuantitiesCritical'
mulitple_exponents_dir = '~/super-resolving-ising/MultipleExponents'
'