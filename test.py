# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 16:25:59 2018

@author: Stavros
"""

import numpy as np
from data.loaders import TestData
from data.model_loader import ModelLoader
from networks.utils import set_GPU_memory, create_directory
from networks.ising import get_observables_with_corr_and_tpf as get_observables
# Returns 12 observables 
# [Mag, En, Susc, specHeat, Mag2, Mag4, En2, 
# tpf(L/2), tpf(L/4), S0, S1, S2]
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-CR', type=bool, default=False, help='critical data')
parser.add_argument('-RGWD', type=bool, default=False, help='well defined RG')
parser.add_argument('-L', type=int, default=16, help='output size')
parser.add_argument('-GPU', type=float, default=0.3, help='GPU memory fraction')

parser.add_argument('-Mind', type=int, default=0, help='model index')
parser.add_argument('-OUT', type=bool, default=False, help='save output')
parser.add_argument('-Tind', type=list, default=[], help='temperatures indices to train')

parser.add_argument('-nTE', type=int, default=100000, help='test samples')
parser.add_argument('-TEST', type=int, default=5000, help='test size')

def calculate_observables(data_or, data_in, data_out, T):
    pred_samp = (data_out > np.random.random(data_out.shape)).astype(np.int)
    
    obs = np.zeros([5, 12])
    obs[0] = get_observables(data_or, T)
    obs[1] = get_observables(data_in, T)
    obs[2] = get_observables(data_out, T)
    obs[3] = get_observables(np.round(data_out), T)
    obs[4] = get_observables(pred_samp, T)
    
    return obs

def main(args):
    data = TestData(args)
    set_GPU_memory(fraction=args.GPU)
    
    model = ModelLoader(list_ind=args.Mind, critical=args.CR)
        
    if args.CR:
        from data.directories import quantities_critical_dir
        
        ## Make predictions ##
        pred_cont = model.graph.predict(data.test_in[:args.TEST])
        
        ## Calculate observables ##
        obs = calculate_observables(data.test_out[:args.TEST,:,:,0], 
                                    data.test_in[:args.TEST,:,:,0], 
                                    pred_cont[:,:,:,0],
                                    T = 2 / np.log(1 + np.sqrt(2)))
        
        ## Save observables ##
        create_directory(quantities_critical_dir)
        np.save(quantities_critical_dir + '/%s_samples%d.npy'%(model.name, 
                                                               args.TEST), obs)
        
        ## Save network output ##
        if args.OUT:
            from data.directories import output_critical_dir
            create_directory(output_critical_dir)
            np.save(output_critical_dir + '/%s_samples%d.npy'%(
                    model.name, args.TEST), pred_cont)
    
    else:
        from data.directories import quantities_dir, T_list
        from data.loaders import temp_partition
                
        if args.OUT:
            from data.directories import output_dir
            create_directory(output_dir + '/' + model.name)
        
        if args.Tind == []:
            args.Tind = np.arange(len(T_list))
        
        obs = np.zeros([len(args.Tind), 5, 12])
        for (iT, T) in enumerate(T_list[args.Tind]):
            ## Update model temperature ##
            model.update_temperature(T=T)
            
            ## Make predictions ##
            data_in = temp_partition(data.test_in, iT, n_samples=args.nTE)
            pred_cont = model.graph.predict(data_in)
            
            ## Calculate observables ##
            obs[iT] = calculate_observables(
                    temp_partition(data.test_out[:,:,:,0], iT, n_samples=args.nTE),
                    data_in[:,:,:,0], pred_cont[:,:,:,0], T=T)
                    
            ## Save network output ##
            if args.OUT:
                np.save(output_dir + '/T%.4f.npy'%T, pred_cont)
                
            print('Temperature %d / %d done!'%(iT+1, len(args.Tind)))
        
        ## Save observables ##
        create_directory(quantities_dir)
        np.save(quantities_dir + '/%s.npy'%model.name, np.array(obs))
        
main(parser.parse_args())