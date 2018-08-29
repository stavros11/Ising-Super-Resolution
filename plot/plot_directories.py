# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:19:31 2018

@author: Stavros
"""

from numpy import linspace

#BASIC_DIR = 'C:/Users/Stavros.SAVVAS-PROBOOK/Documents/Scripts_and_Programs/SR_results/'
#BASIC_DIR = 'C:/Users/Stavros/Documents/Scripts_and_programs/SR_results/'
#BASIC_DIR = 'C:/Users/Stavros.SAVVAS-PROBOOK/Documents/Scripts_and_Programs/SR_results/'
BASIC_DIR = 'D:/Ising-Super-Resolution-Data/SR_results/'

## Temperature list ##
T_list = linspace(0.01, 4.538, 32)
T_list1D = linspace(0.01, 3.515, 32)

## Network directories ##
models_dir = BASIC_DIR + 'Models'
models_critical_dir = BASIC_DIR + 'ModelsCritical'
metrics_save_dir = BASIC_DIR + 'Metrics'
metrics_critical_save_dir = BASIC_DIR + 'MetricsCritical'

## Quantities directories ##
quantities_dir = BASIC_DIR + 'Quantities'
quantities_real_dir = BASIC_DIR + 'QuantitiesReal'
quantities_critical_dir = BASIC_DIR + 'QuantitiesCritical'
multiple_exponents_dir = BASIC_DIR + 'MultipleExponents'
seaborn_dir = BASIC_DIR + 'SeabornQuantities'

quantities_dir1D = BASIC_DIR + 'Quantities1D'
quantities_dir1D_fixed = BASIC_DIR + 'Quantities1D_Fixed'
quantities_dir1D_rep = BASIC_DIR + 'QuantitiesRep1D'

## Output directories ##
output_dir = BASIC_DIR + 'Output'