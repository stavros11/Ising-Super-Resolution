# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 12:01:08 2018

@author: Stavros
"""

import numpy as np
from data.loaders import TrainingData
from data.directories import models_critical_save_dir, metrics_critical_save_dir
from data.directories import multiple_exponents_dir
from networks.utils import set_GPU_memory
from networks.train import TrainerCritical, create_directory
from networks.consecutive import upsampling
from argparse import ArgumentParser
from scipy.stats import linregress

parser = ArgumentParser()
parser.add_argument('-C', type=int, default=1, help='number of calculations')
parser.add_argument('-UP', type=int, default=3, help='number of upsamplings')
parser.add_argument('-PRreg', type=bool, default=True, help='print regression')
parser.add_argument('-TPF', type=bool, default=True, help='calculate two point function')
parser.add_argument('-CORR', type=bool, default=True, help='calculate correlation length')

parser.add_argument('-GPU', type=float, default=0.4, help='GPU memory fraction')
parser.add_argument('-RGWD', type=bool, default=False, help='well defined RG')
parser.add_argument('-L', type=int, default=16, help='output size')
parser.add_argument('-nTR', type=int, default=40000, help='train samples')
parser.add_argument('-nTE', type=int, default=100000, help='test samples')
parser.add_argument('-TRS', type=int, default=40000, help='train size')
parser.add_argument('-VALS', type=int, default=3000, help='validation size')
parser.add_argument('-VER', type=int, default=1, help='version for name')

parser.add_argument('-PBC', type=bool, default=True, help='use PBC padding')
parser.add_argument('-ACT', type=str, default='relu', help='hidden activation')
parser.add_argument('-HF', nargs='+', type=int, default=None, help='hidden filters list')
parser.add_argument('-K', nargs='+', type=int, default=None, help='kernels list')

parser.add_argument('-OPT', type=str, default='Adam', help='optimizer')
parser.add_argument('-CE', type=bool, default=True, help='use cross-entropy')
parser.add_argument('-magR', type=float, default=0.0, help='magnetization regularization')
parser.add_argument('-enR', type=float, default=0.0, help='energy regularization')

parser.add_argument('-EP', type=int, default=100, help='epochs')
parser.add_argument('-VB', type=int, default=0, help='verbose')
parser.add_argument('-BS', type=int, default=1000, help='batch size')
parser.add_argument('-ES', type=bool, default=False, help='early stopping')
parser.add_argument('-ESpat', type=int, default=50, help='early stopping patience')
parser.add_argument('-ESdelta', type=float, default=0.001, help='early stopping delta')

parser.add_argument('-CB', type=bool, default=False, help='use batches for calculation')
parser.add_argument('-CBS', type=int, default=1000, help='calculation batches')
parser.add_argument('-NUP', type=int, default=4, help='maximum upsampling number without batches')

def main(args):
    args.CR = True
    args.model_dir = models_critical_save_dir
    args.metrics_dir= metrics_critical_save_dir
    set_GPU_memory(fraction=args.GPU)
    
    ### HF, K default values ###
    if args.HF == None:
        args.HF = [64, 32]
    if args.K == None:
        args.K = [5, 1, 3]
    
    if args.PRreg:
        L0 = int(np.log2(args.L))
        L_list = 2**np.arange(L0, L0+1+args.UP)
        
    if args.CB:
        from networks.consecutive import upsampling_batches as upsampling
    else:
        from networks.consecutive import upsampling
    
    data = TrainingData(args)
    trainer = TrainerCritical(args)
    observables = []
    for iC in range(args.C):
        trainer.train(data, run_time=iC)
        obs = upsampling(data.test_out, trainer.model, args)
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
    np.save('%s/%s_C%dUP%dVER%d.npy'%(multiple_exponents_dir, trainer.name,
                             args.C, args.UP, args.VER), np.array(observables))


main(parser.parse_args())
