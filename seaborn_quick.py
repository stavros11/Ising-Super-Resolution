# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:29:39 2018

@author: Admin
"""

import numpy as np
from data.loaders import read_file, temp_partition, add_index
from data.directories import seaborn_dir, T_list
from data.model_loader import ModelLoader
from networks.ising import Ising
from networks.utils import set_GPU_memory, create_directory
from networks.architectures import duplicate_simple2D_pbc as duplicate
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('-RGWD', type=bool, default=True, help='well defined RG')
parser.add_argument('-L', type=int, default=32, help='output size')
parser.add_argument('-GPU', type=float, default=0.2, help='GPU memory fraction')
parser.add_argument('-iT', type=int, default=15, help='temperature index')

parser.add_argument('-Mind', type=int, default=0, help='model index')
parser.add_argument('-PBC', type=bool, default=False, help='use PBC padding')
parser.add_argument('-ACT', type=str, default='relu', help='hidden activation')
parser.add_argument('-HF', nargs='+', type=int, default=None, help='hidden filters list')
parser.add_argument('-K', nargs='+', type=int, default=None, help='kernels list')

parser.add_argument('-nTE', type=int, default=10000, help='test samples')
parser.add_argument('-TEST', type=int, default=10000, help='test size')

## Default ##
parser.add_argument('-CR', type=bool, default=False, help='critical data')


def main(args):    
    ## Renormalized temperature (inverse)    
    T_ren_inv = np.array([0.01, 0.01, 0.01, 0.01, 0.01,
       1.21835191, 1.22976684, 1.39674347, 1.51484435, 1.65761354,
       1.75902208, 1.85837041, 1.95260925, 2.07132396, 2.13716533,
       2.25437054, 2.29606717, 2.38018868, 2.44845189, 2.51316151,
       2.58725426, 2.6448879 , 2.7110948 , 2.74426717, 2.81525268,
       2.87031377, 2.90806294, 2.98742994, 3.03780331, 3.10501399,
       3.17323991, 3.19663683])
    
    ## Read data ##
    L_list = [args.L, args.L//2, args.L]
    data_or = temp_partition(add_index(read_file(L=args.L, n_samples=args.nTE)), args.iT, n_samples=args.nTE)
    data_in = temp_partition(add_index(read_file(L=args.L//2, n_samples=args.nTE)), args.iT, n_samples=args.nTE)
    
    ## Set model ##
    set_GPU_memory(fraction=args.GPU)
    model = ModelLoader(list_ind=args.Mind, critical=args.CR)
    print('\nModel %s loaded.'%model.name)
    print('Temperature = %.4f\n'%T_list[args.iT])

    if args.TEST > args.nTE:
        args.TEST = args.nTE
    
    ## Find transformed temperatures ##
    Tr = T_ren_inv[args.iT]
        
    ## Find closer value from T_list to update model temperature ##
    iT_closer = np.abs(T_list - Tr).argmin()
    model.update_temperature(T=T_list[iT_closer])
    extrapolated_model = duplicate(model.graph, data_in.shape,
                                   hid_filters=args.HF, kernels=args.K, hid_act=args.ACT)
    
    print('\nModel %s loaded.'%model.name)
    print('Temperature = %.4f'%T_list[args.iT])
    print('Renormalized temperatute = %.4f\n'%T_list[iT_closer])
    
    ## Make predictions ##
    pred_cont = extrapolated_model.predict(data_in)
    
    ## Calculate observables ##
    data_list = [temp_partition(read_file(L=args.L, n_samples=args.nTE), iT_closer),
                 temp_partition(read_file(L=args.L//2, n_samples=args.nTE), iT_closer),
                 pred_cont[:,:,:,0] > np.random.random(pred_cont.shape[:-1])]
    
    obj_list = [Ising(2 * x - 1) for x in data_list]
    obs = np.zeros([3, 2, args.nTE])
    for (i, obj) in enumerate(obj_list):
        obj._calculate_magnetization()
        obj._calculate_energy()
        obs[i, 0] = np.abs(obj.sample_mag) * 1.0 / L_list[i]**2
        obs[i, 1] = obj.sample_energy * 1.0 / L_list[i]**2
        
    create_directory(seaborn_dir)
    np.save(seaborn_dir + '/%s_extr_iT%d.npy'%(model.name, args.iT), np.array(obs))
    
main(parser.parse_args())