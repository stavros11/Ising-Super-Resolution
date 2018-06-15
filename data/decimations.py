# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 21:51:16 2018

@author: Stavros
"""

import numpy as np

##########################################
########## DECIMATION FUNCTIONS ##########
##########################################
    
def block_rg(state):
    (n_samples, Ly, Lx) = state.shape
    # Calculate sums of neighboring cells in x direction
    sumx = state[:,:,1:] + state[:,:,:Lx-1]
    # Truncate by keeping only the even terms that belong to the same renormalization block
    sumx_trunc = sumx[:,:,np.arange(0,Lx,2)]
        
    #Same for the y direction by starting from the truncated state
    sumy = sumx_trunc[:,1:,:] + sumx_trunc[:,:Ly-1,:]
    sumy_trunc = sumy[:,np.arange(0,Ly,2),:]
    # Now we have a renormalized state where each site corresponds to a former block and 
    # it has the value of the block's sum
   
    # Convert the sum value to a spin value
    # If sum>2 --> spin 1, sum<2 --> spin 0, sum=2 --> Probabilities
    # To do this add small noise (symmetric around zero)
    noise = 0.2 * np.random.random([n_samples, np.int(Ly/2), np.int(Lx/2)]) - 0.1
    return ( sumy_trunc + noise > 2 ).astype(np.int)

def block_rg_WD(state):
    ### Same as block_rg but this is not random:
    ### The 2-2 blocks get the value of the upper left spin
    (n_samples, Ly, Lx) = state.shape
    sumx = state[:,:,1:] + state[:,:,:Lx-1]
    sumx_trunc = sumx[:,:,np.arange(0,Lx,2)]
    sumy = sumx_trunc[:,1:,:] + sumx_trunc[:,:Ly-1,:]
    sumy_trunc = sumy[:,np.arange(0,Ly,2),:]
    
    noise = 0.2 * state[:, np.arange(0,Ly,2),:]
    noise = noise[:,:,np.arange(0,Lx,2)]
    
    return ( sumy_trunc + noise > 2 ).astype(np.int)

def block_sum(state):
    (n_samples, Ly, Lx) = state.shape
    ### Calculate sums (part of the block RG function)
    ### Useful to test some heuristics
    sumx = state[:,:,1:] + state[:,:,:-1]
    # Truncate by keeping only the even terms that belong to the same block
    sumx_trunc = sumx[:,:,np.arange(0,Lx,2)]       
    #Same for the y direction by starting from the truncated state
    sumy = sumx_trunc[:,1:,:] + sumx_trunc[:,:-1,:]
    return sumy[:,np.arange(0,Ly,2),:]