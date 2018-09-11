# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:19:11 2018

@author: Stavros
"""

import numpy as np
from data.directories import quantities_real_dir, T_list
from data.loaders import read_file, add_index, temp_partition
from data.model_loader import ModelLoader
from networks.utils import set_GPU_memory, create_directory, calculate_observables_real
from networks.architectures import duplicate_simple1D_pbc
# Returns 7 observables 
# [Mag, En, Susc, specHeat, Mag2, Mag4, En2]
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-L', type=int, default=64, help='output size')
parser.add_argument('-GPU', type=float, default=0.2, help='GPU memory fraction')

parser.add_argument('-Mind', type=int, default=0, help='model index')
parser.add_argument('-ACT', type=str, default='relu', help='hidden activation')
parser.add_argument('-HF', nargs='+', type=int, default=None, help='hidden filters list')
parser.add_argument('-K', nargs='+', type=int, default=None, help='kernels list')

parser.add_argument('-nTE', type=int, default=10000, help='test samples')
parser.add_argument('-TEST', type=int, default=10000, help='test size')

def main(args):
    mc_small = read_file(L=args.L//2, n_samples=args.nTE)
    mc_large = read_file(L=args.L, n_samples=args.nTE)
    
    set_GPU_memory(fraction=args.GPU)
    model = ModelLoader(list_ind=args.Mind)
    
    if args.TEST > args.nTE:
        args.TEST = args.nTE
        
    obs = np.zeros([len(T_list), 5, 7])
    
    ## First upsampling
    T_ren = 2.0 / np.arccosh(np.exp(2.0 / T_list))
    for (iT, T) in enumerate(T_ren):
        ## Update model temperature ##
        T_model = T_list[np.abs(T - T_list).argmin()]
        model.update_temperature(T=T_model)
        
        ## Make predictions ##
        data_in = temp_partition(mc_small, iT, n_samples=args.nTE)
        model_large = duplicate_simple1D_pbc(model.graph, data_in.shape,
                                             hid_filters=args.HF, kernels=args.K)
        pred = model_large.predict(add_index(data_in))
        
        ## Calculate observables ##
        obs[iT] = calculate_observables_real(temp_partition(mc_large, iT, n_samples=args.nTE),
           data_in, pred[:,:,0], T=T_list[iT], Tr=T)
        
        print('Temperature %d / %d done!'%(iT+1, len(T_list)))
        
    ## Save observables ##
    create_directory(quantities_real_dir)
    np.save(quantities_real_dir + '/%s.npy'%(model.name), obs)
        
main(parser.parse_args())