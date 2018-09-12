# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:55:44 2018

@author: Admin
"""

import numpy as np
from data.loaders import read_file, temp_partition, add_index
from data.directories import quantities_real_dir, T_list
from data.model_loader import ModelLoader
from networks.utils import set_GPU_memory, create_directory, calculate_observables_real
from argparse import ArgumentParser

# Returns 12 observables 
# [Mag, En, Susc, specHeat, Mag2, Mag4, En2, 
# tpf(L/2), tpf(L/4), S0, S1, S2]

parser = ArgumentParser()
parser.add_argument('-RGWD', type=bool, default=True, help='well defined RG')
parser.add_argument('-L', type=int, default=32, help='output size')
parser.add_argument('-GPU', type=float, default=0.3, help='GPU memory fraction')

parser.add_argument('-Mind', type=int, default=0, help='model index')
parser.add_argument('-Tind', nargs='+', type=int, default=None, help='temperatures indices to train')
parser.add_argument('-OUT', type=bool, default=False, help='save output')

parser.add_argument('-PBC', type=bool, default=False, help='use PBC padding')
parser.add_argument('-ACT', type=str, default='relu', help='hidden activation')
parser.add_argument('-HF', nargs='+', type=int, default=None, help='hidden filters list')
parser.add_argument('-K', nargs='+', type=int, default=None, help='kernels list')

parser.add_argument('-nTE', type=int, default=10000, help='test samples')
parser.add_argument('-TEST', type=int, default=10000, help='test size')

## Default ##
parser.add_argument('-CR', type=bool, default=False, help='critical data')

def main(args):
    if args.PBC:
        from networks.architectures import duplicate_simple2D_pbc as duplicate
    else:
        from networks.architectures import duplicate_simple2D as duplicate
    
    ## Renormalized temperature (inverse)    
    T_ren_inv = np.array([0.01, 0.01, 0.01, 0.01, 0.01,
       1.21835191, 1.22976684, 1.39674347, 1.51484435, 1.65761354,
       1.75902208, 1.85837041, 1.95260925, 2.07132396, 2.13716533,
       2.25437054, 2.29606717, 2.38018868, 2.44845189, 2.51316151,
       2.58725426, 2.6448879 , 2.7110948 , 2.74426717, 2.81525268,
       2.87031377, 2.90806294, 2.98742994, 3.03780331, 3.10501399,
       3.17323991, 3.19663683])
    
    ## Read data ##
    data_or = read_file(L=args.L, n_samples=args.nTE)
    data_in = add_index(read_file(L=args.L//2, n_samples=args.nTE))
    
    ## Set model ##
    set_GPU_memory(fraction=args.GPU)
    model = ModelLoader(list_ind=args.Mind, critical=args.CR)

    if args.TEST > args.nTE:
        args.TEST = args.nTE

    if args.OUT:
        from data.directories import output_dir
        create_directory(output_dir + '/' + model.name)
    
    if args.Tind == None:
        args.Tind = np.arange(len(T_list))
    
    obs = np.zeros([len(args.Tind), 5, 12])
    for (iT, T) in enumerate(T_list[args.Tind]):
        ## Find transformed temperatures ##
        Tr = T_ren_inv[iT]
        
        ## Find closer value from T_list to update model temperature ##
        T_closer = T_list[np.abs(T_list - Tr).argmin()]
        model.update_temperature(T=T_closer)
        extrapolated_model = duplicate(model.graph, data_in.shape,
                                       hid_filters=args.HF, kernels=args.K, hid_act=args.ACT)
        
        ## Make predictions ##
        data_in_T = temp_partition(data_in, iT, n_samples=args.nTE)
        pred_cont = extrapolated_model.predict(data_in_T)
        
        ## Calculate observables ##
        if iT > 5:
            Tr_calc = Tr
        else:
            Tr_calc = T
        obs[iT] = calculate_observables_real(temp_partition(data_or, iT, n_samples=args.nTE), 
           data_in_T[:,:,:,0], pred_cont[:,:,:,0], T=T, Tr=Tr_calc)
                    
        ## Save network output ##
        if args.OUT:
            np.save(output_dir + '/%s/T%.4f.npy'%(model.name, T), pred_cont)
        
        print('Temperature %d / %d done!'%(iT+1, len(args.Tind)))
        
    ## Save observables ##
    create_directory(quantities_real_dir)
    np.save(quantities_real_dir + '/%s_extr.npy'%model.name, np.array(obs))
        
main(parser.parse_args())