# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:30:34 2018

@author: Stavros
"""

import numpy as np
from data.loaders import read_file, temp_partition, add_index
from data.directories import quantities_real_dir, T_list
from data.model_loader import ModelLoader
from networks.utils import set_GPU_memory, create_directory, calculate_observables
from renormalization.curves import inv_curve
from argparse import ArgumentParser

# Returns 12 observables 
# [Mag, En, Susc, specHeat, Mag2, Mag4, En2, 
# tpf(L/2), tpf(L/4), S0, S1, S2]

parser = ArgumentParser()
parser.add_argument('-RGWD', type=bool, default=True, help='well defined RG')
parser.add_argument('-L', type=int, default=16, help='output size')
parser.add_argument('-GPU', type=float, default=0.3, help='GPU memory fraction')

parser.add_argument('-Mind', type=int, default=0, help='model index')
parser.add_argument('-Tind', nargs='+', type=int, default=None, help='temperatures indices to train')
parser.add_argument('-OUT', type=bool, default=False, help='save output')

parser.add_argument('-nTE', type=int, default=10000, help='test samples')
parser.add_argument('-TEST', type=int, default=10000, help='test size')

## Default ##
parser.add_argument('-CR', type=bool, default=False, help='critical data')

def main(args):
    ## Load renormalization curve parameters ##
    a_mag, b_mag = np.load('renormalization/Magnetization_Transformation_Params_L%d.npy'%args.L)
    a_en, b_en = np.load('renormalization/Energy_Transformation_Params_L%d.npy'%args.L)
    
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
        Tr_mag = inv_curve(T, a=a_mag, b=b_mag)
        Tr_en = inv_curve(T, a=a_en, b=b_en)
        
        ## Find closer value from T_list to update model temperature ##
        difs = (T_list - Tr_mag)**2 + (T_list - Tr_en)**2
        T_closer = T_list[difs.argmin()]
        model.update_temperature(T=T_closer)
        
        ## Make predictions ##
        data_in_T = temp_partition(data_in, iT, n_samples=args.nTE)
        pred_cont = model.graph.predict(data_in_T)
        
        ## Calculate observables ##
        obs[iT] = calculate_observables(temp_partition(data_or, iT, n_samples=args.nTE), 
           data_in_T[:,:,:,0], pred_cont[:,:,:,0], T=T)#, Tr=(Tr_mag + Tr_en)/2.0)
                    
        ## Save network output ##
        if args.OUT:
            np.save(output_dir + '/%s/T%.4f.npy'%(model.name, T), pred_cont)
        
        print('Temperature %d / %d done!'%(iT+1, len(args.Tind)))
        
        ## Save observables ##
        create_directory(quantities_real_dir)
        np.save(quantities_real_dir + '/%s.npy'%model.name, np.array(obs))
        
main(parser.parse_args())
