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
from networks.ising import two_point_function
from networks.architectures import make_prediction
# Returns 7 observables 
# [Mag, En, Susc, specHeat, Mag2, Mag4, En2]
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-L', type=int, default=32, help='output size')
parser.add_argument('-UP', type=int, default=3, help='number of upsamplings')
parser.add_argument('-TS', type=int, default=5, help='temperature index to start sampling')
parser.add_argument('-TPFD', type=int, default=16, help='divide L for two point function calculation')
parser.add_argument('-GPU', type=float, default=0.3, help='GPU memory fraction')

parser.add_argument('-Mind', type=int, default=0, help='model index')
parser.add_argument('-ACT', type=str, default='relu', help='hidden activation')
parser.add_argument('-HF', nargs='+', type=int, default=None, help='hidden filters list')
parser.add_argument('-K', nargs='+', type=int, default=None, help='kernels list')

parser.add_argument('-nTE', type=int, default=10000, help='test samples')
parser.add_argument('-TEST', type=int, default=10000, help='test size')

def main(args):
    data_mc = read_file(L=args.L, n_samples=args.nTE)
    
    set_GPU_memory(fraction=args.GPU)
    model = ModelLoader(list_ind=args.Mind)
    
    if args.TEST > args.nTE:
        args.TEST = args.nTE
        
    tpf = np.zeros([args.UP, len(T_list)])
    obs = np.zeros([args.UP, len(T_list), 3, 7])
    ## WARNING: Does not contain original MC observables
    
    pred_cont = []
    ## First upsampling
    T_ren = 2.0 / np.arccosh(np.exp(2.0 / T_list))
    for (iT, T) in enumerate(T_ren):
        ## Update model temperature ##
        T_model = T_list[np.abs(T - T_list).argmin()]
        model.update_temperature(T=T_model)
                
        ## Make predictions ##
        pred_cont.append(make_prediction(
                data_in=add_index(temp_partition(data_mc, iT, n_samples=args.nTE)),
                graph=model.graph, hid_filters=args.HF, kernels=args.K, hid_act=args.ACT))
                
                
        ## Calculate observables ##
        obs[0, iT] = calculate_observables_rep(pred_cont[iT][:,:,0], Tr=T)
        tpf[0, iT] = two_point_function(2 * pred_cont[iT][:,:,0] - 1, k=pred_cont[iT].shape[1] // args.TPFD)
        
        print('Temperature %d / %d done!'%(iT+1, len(T_list)))
        
    print('\nUpsampling 1 / %d completed!\n'%args.UP)
    
    for iUP in range(1, args.UP):
        T_ren = 2.0 / np.arccosh(np.exp(2.0 / T_ren))
        for (iT, T) in enumerate(T_ren):
            T_model = T_list[np.abs(T - T_list).argmin()]
            model.update_temperature(T=T_model)
                    
            ## Make predictions ##
            if iT < args.TS:
                sampled_in = np.round(pred_cont[iT])
            else:
                sampled_in = (pred_cont[iT] > np.random.random(pred_cont[iT].shape)).astype(np.int)
            
            pred_cont[iT] = make_prediction(data_in=sampled_in, graph=model.graph, 
                     hid_filters=args.HF, kernels=args.K, hid_act=args.ACT)

            ## Calculate observables ##
            obs[iUP, iT] = calculate_observables_rep(pred_cont[iT][:,:,0], Tr=T)
            tpf[iUP, iT] = two_point_function(2 * pred_cont[iT][:,:,0] - 1, k=pred_cont[iT].shape[1] // args.TPFD)
            
            print('Temperature %d / %d done!'%(iT+1, len(T_list)))
        
        print('\nUpsampling %d / %d completed!\n'%(iUP+1, args.UP))
            
    ## Save observables ##
    create_directory(quantities_rep_dir)
    np.save(quantities_rep_dir + '/%s_TS%d.npy'%(model.name, args.TS), obs)
    np.save(quantities_rep_dir + '/%s_TS%d_TPFk%d.npy'%(model.name, args.TS, args.TPFD), tpf)
        
main(parser.parse_args())
