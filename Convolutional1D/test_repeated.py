# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:49:59 2018

@author: User
"""

import numpy as np
from data.directories import quantities_rep_dir, T_list
from data.loaders import read_file, add_index, temp_partition
from data.model_loader import ModelLoader
from networks.utils import set_GPU_memory, create_directory, calculate_observables_rep
# Returns 7 observables 
# [Mag, En, Susc, specHeat, Mag2, Mag4, En2]
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-L', type=int, default=16, help='output size')
parser.add_argument('-UP', type=int, default=3, help='number of upsamplings')
parser.add_argument('-GPU', type=float, default=0.3, help='GPU memory fraction')

parser.add_argument('-Mind', type=int, default=0, help='model index')

parser.add_argument('-nTE', type=int, default=10000, help='test samples')
parser.add_argument('-TEST', type=int, default=10000, help='test size')

def main(args):
    data_mc = read_file(L=args.L, n_samples=args.nTE)
    
    set_GPU_memory(fraction=args.GPU)
    model = ModelLoader(list_ind=args.Mind)
    
    if args.TEST > args.nTE:
        args.TEST = args.nTE
        
    obs = np.zeros([args.UP, len(args.Tind), 3, 7])
    ## WARNING: Does not contain original MC observables
    
    ## First upsampling
    T_ren = 2.0 / np.arccosh(np.exp(2.0 / T_list))
    for (iT, T) in enumerate(T_ren):
        ## Update model temperature ##
        T_model = T_list[np.abs(T - T_list).argmin()]
        model.update_temperature(T=T_model)
                
        ## Make predictions ##
        pred_cont = model.graph.predict(temp_partition(data_mc, iT, n_samples=args.nTE))
                
        ## Calculate observables ##
        obs[iT] = calculate_observables(
                temp_partition(data.test_out[:,:,:,0], iT, n_samples=args.nTE),
                data_in[:,:,:,0], pred_cont[:,:,:,0], T=T)
                                       
        print('Temperature %d / %d done!'%(iT+1, len(args.Tind)))
    
    for iUP in range(args.UP):
        for (iT, T) in enumerate(T_list):
            ## Update model temperature ##
            model.update_temperature(T=T)
                
            ## Make predictions ##
            data_in = temp_partition(data.test_in, iT, n_samples=args.nTE)
            pred_cont = model.graph.predict(data_in)
                
            ## Calculate observables ##
            obs[iT] = calculate_observables(
                    temp_partition(data.test_out[:,:,:,0], iT, n_samples=args.nTE),
                    data_in[:,:,:,0], pred_cont[:,:,:,0], T=T)
                                       
            print('Temperature %d / %d done!'%(iT+1, len(args.Tind)))
        
    ## Save observables ##
    create_directory(quantities_rep_dir)
    np.save(quantities_rep_dir + '/%s.npy'%model.name, np.array(obs))
        
main(parser.parse_args())