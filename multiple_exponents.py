# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 18:59:45 2018

@author: Stavros
"""

import os
import numpy as np
from data.loaders import read_file_GPU, add_index
from data.directories import multiple_exponents_test_dir
from data.model_loader import critical_model_from_file
from networks.utils import set_GPU_memory, create_directory
from networks.consecutive import upsampling
from argparse import ArgumentParser
from scipy.stats import linregress

parser = ArgumentParser()
#parser.add_argument('-C', type=int, default=1, help='number of calculations')
parser.add_argument('-UP', type=int, default=3, help='number of upsamplings')
parser.add_argument('-PRreg', type=bool, default=True, help='print regression')
parser.add_argument('-TPF', type=bool, default=True, help='calculate two point function')
parser.add_argument('-CORR', type=bool, default=True, help='calculate correlation length')

parser.add_argument('-NET', type=int, default=0, help='network choice')
## These should be consistent with selected model ##
parser.add_argument('-PBC', type=bool, default=True, help='use PBC padding')
parser.add_argument('-ACT', type=str, default='relu', help='hidden activation')
parser.add_argument('-HF', nargs='+', type=int, default=None, help='hidden filters list')
parser.add_argument('-K', nargs='+', type=int, default=None, help='kernels list')

parser.add_argument('-GPU', type=float, default=0.3, help='GPU memory fraction')
parser.add_argument('-L', type=int, default=16, help='output size')
parser.add_argument('-nTE', type=int, default=100000, help='test samples')
parser.add_argument('-VER', type=int, default=1, help='version for name')

args = parser.parse_args()
args.CR = True

from data.directories import models_critical_save_dir as basic_dir
name = [o for o in os.listdir(basic_dir) 
if os.path.isdir(os.path.join(basic_dir,o))][args.NET]

set_GPU_memory(fraction=args.GPU)
if args.PRreg:
    L0 = int(np.log2(args.L))
    L_list = 2**np.arange(L0, L0+1+args.UP)

## Load model ##
#model = ModelLoader(args.NET, critical=True)

data = add_index(read_file_GPU(L=args.L))
#data = add_index(read_file_critical(L=args.L, n_samples=args.nTE))
observables = []
for model_name in os.listdir(os.path.join(basic_dir, name)):
    print(data.shape)
    ## Load model ##
    model = critical_model_from_file(os.path.join(basic_dir, name, model_name))
    print('\n%s\n'%model_name)
    
    obs = upsampling(data, model, args)
    observables.append(obs)
    
    if args.PRreg:
        print('Beta:')
        print(linregress(np.log10(L_list), np.log10(obs[0])))
        print('Gamma:')
        print(linregress(np.log10(L_list), np.log10(obs[2])))
        if args.TPF:
            print('Eta1:')
            print(linregress(np.log10(L_list/2.0), np.log10(obs[7])))
            print('Eta2:')
            print(linregress(np.log10(L_list/4.0), np.log10(obs[8])))

create_directory(multiple_exponents_test_dir)
np.save('%s/%s_C%dUP%d.npy'%(multiple_exponents_test_dir, name,
                             len(observables), args.UP), np.array(observables))