# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:18:23 2018

@author: Stavros
"""

from networks.utils import set_GPU_memory
from data.loaders import TrainingData
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-CR', type=bool, default=False, help='critical data')
parser.add_argument('-RGWD', type=bool, default=False, help='well defined RG')
parser.add_argument('-L', type=int, default=16, help='output size')
parser.add_argument('-Tind', nargs='+', type=int, default=None, help='temperatures indices to train')

parser.add_argument('-GPU', type=float, default=0.4, help='GPU memory fraction')
parser.add_argument('-nTR', type=int, default=10000, help='train samples')
parser.add_argument('-nTE', type=int, default=10000, help='test samples')
parser.add_argument('-TRS', type=int, default=10000, help='train size')
parser.add_argument('-VALS', type=int, default=2000, help='validation size')
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
parser.add_argument('-BS', type=int, default=1000, help='batch size')
parser.add_argument('-ES', type=bool, default=False, help='early stopping')
parser.add_argument('-ESpat', type=int, default=50, help='early stopping patience')
parser.add_argument('-ESdelta', type=float, default=0.0001, help='early stopping delta')
parser.add_argument('-VB', type=int, default=0, help='verbose')

def main(args):
    set_GPU_memory(fraction=args.GPU)
    
    ### HF, K default values ###
    if args.HF == None:
        args.HF = [64, 32]
    if args.K == None:
        args.K = [5, 1, 3]
    
    ### Check data sizes ###
    if args.TRS > args.nTR:
        args.TRS = args.nTR
    if args.VALS > args.nTE:
        args.VALS = args.nTE
    
    print('Hidden Filters: ' + str(args.HF))
    print('Kernels: ' + str(args.K))
    
    data = TrainingData(args)
    if args.CR:
        from data.directories import models_critical_save_dir, metrics_critical_save_dir
        from networks.train import TrainerCritical
        args.model_dir = models_critical_save_dir
        args.metrics_dir= metrics_critical_save_dir
        
        trainer = TrainerCritical(args)
        trainer.compiler(data)
        
    else:
        from data.directories import models_save_dir, metrics_save_dir, T_list
        from networks.train import TrainerTemp
        args.model_dir = models_save_dir
        args.metrics_dir = metrics_save_dir
        
        if args.Tind == None:
            args.Tind = range(len(T_list))
        
        args.T_list = T_list[args.Tind]
        
        trainer = TrainerTemp(args)
    
    trainer.train(data)
    return
    
main(parser.parse_args())
