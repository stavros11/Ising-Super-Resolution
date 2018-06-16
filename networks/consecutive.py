# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 12:34:19 2018

@author: Stavros
"""

import numpy as np

def create_obs_function(args):
    if args.TPF and args.CORR:
        from ising import get_observables_with_corr_and_tpf as f
        return f
    if args.TPF:
        from ising import get_observables_with_tpf as f
        return f
    from ising import get_observables as f
    return f

def upsampling(init_data, model, args):
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    if args.PBC:
        from architectures import duplicate_simple2D_pbc
        def duplication_function(old_model, x):
            return duplicate_simple2D_pbc(old_model, x,
                                          hid_filters=args.HF,
                                          kernels=args.K,
                                          hid_act=args.ACT)
    else:
        from architectures import duplicate_simple2D
        def duplication_function(old_model, x):
            return duplicate_simple2D(old_model, x,
                                      hid_filters=args.HF,
                                      kernels=args.K,
                                      hid_act=args.ACT)
    
    get_observables = create_obs_function(args)

    # Number of args that get_observables function returns: 7
    obs = [get_observables(init_data[:,:,:,0], Tc)]
    
    state = np.copy(init_data)
    for i in range(args.UP):
        model = duplication_function(model, state.shape)
        cont_state = model.predict(state)
        
        state = (cont_state > np.random.random(cont_state.shape)).astype(np.int)
        
        obs.append(get_observables(state[:,:,:,0], Tc))
        print('%d / %d upsamplings done!'%(i+1, args.UP))
        
    return np.array(obs).T