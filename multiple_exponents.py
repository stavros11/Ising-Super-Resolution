# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 18:59:45 2018

@author: Stavros
"""

import numpy as np
from data.loaders import TrainingData
from data.directories import models_critical_save_dir
from data.directories import multiple_exponents_dir
from networks.utils import set_GPU_memory, load_model_from_list
from networks.train import create_directory
from networks.consecutive import upsampling
from argparse import ArgumentParser
from scipy.stats import linregress

parser = ArgumentParser()
parser.add_argument('-C', type=int, default=1, help='number of calculations')
parser.add_argument('-UP', type=int, default=3, help='number of upsamplings')
parser.add_argument('-PRreg', type=bool, default=True, help='print regression')
parser.add_argument('-TPF', type=bool, default=False, help='calculate two point function')
parser.add_argument('-CORR', type=bool, default=False, help='calculate correlation length')

parser.add_argument('-NET', type=int, default=1, help='network choice')
parser.add_argument('-GPUF', type=float, default=0.4, help='GPU memory fraction')
parser.add_argument('-L', type=int, default=16, help='output size')
parser.add_argument('-nTR', type=int, default=40000, help='train samples')
parser.add_argument('-nTE', type=int, default=100000, help='test samples')
parser.add_argument('-VER', type=int, default=1, help='version for name')

args = parser.parse_args()
args.CR = True


set_GPU_memory(fraction=args.GPUF)
if args.PRreg:
    L0 = int(np.log2(args.L))
    L_list = 2**np.arange(L0, L0+1+args.UP)

## Load model ##
name, model = load_model_from_list(models_critical_save_dir, args.NET)

data = TrainingData(args)
observables = []
for iC in range(args.C):
    obs = upsampling(data.test_out, model, args)
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

create_directory(multiple_exponents_dir)
np.save('%s/%s_C%dUP%d.npy'%(multiple_exponents_dir, name,
                             args.C, args.UP), np.array(observables))