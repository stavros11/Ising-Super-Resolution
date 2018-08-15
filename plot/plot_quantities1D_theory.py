# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:18:06 2018

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 14})

from plot_directories import T_list1D as T_list
from plot_directories import quantities_dir1D, quantities_dir1D_rep
# Use this T_list when plot_directories module is not available
#T_list = np.linspace(0.01, 3.515, 32)

T_list_th = np.linspace(T_list[0], T_list[-1], 1000)

def energy_theory(T, J=1, N=32):
    th = np.tanh(J / T)
    thN_1 = th ** (N-1)
    ch2 = np.cosh(J / T) ** 2
    
    E = thN_1 / (ch2 * (1 + thN_1 * th))
    return - J * (th + E)

def specHeat_theory(T, J=1, N=32):
    sh2 = 1 / np.cosh(J/T)**2
    th = np.tanh(J/T)
    denom = 1 + th**N
    
    C1 = ((N-1) * th**(N-2) - th**(2*N-2)) / denom
    C2 = 2 / denom - 1
    
    return J * (sh2 * C1 + C2) * sh2 / T**2

### !!! .NPY DESCRIPTION !!! ###
# obs = (32, 5, 12)
# ind1: temperatures
# ind2: [MC, RG, SR continuous, SR rounded, SR sampled]
# ind3: [Mag, En, Susc, specHeat, Mag2, Mag4, En2, tpf(L/2), 
#        tpf(L/4), S0, S1, S2]

# Load data (fix .npy directory here)
NAME = 'Simple1D32relu_L1_32_K53_PBC_MReg0.00EReg0.10B1000_TS6'
obs_or = np.load('%s/%s.npy'%(quantities_dir1D, NAME[:-4]))
obs_rep = np.load('%s/%s.npy'%(quantities_dir1D_rep, NAME))

# Use rounding instead of sampling for the five lowest temperatures 
# to correct noise in susc and Cv
n_correct = 6
obs_or[:n_correct, -1] = obs_or[:n_correct, -2]
obs_rep[:, :n_correct, -1] = obs_rep[:, :n_correct, -2]

## Manually correct Cv (quickly - ERASE THIS)
for i in range(32):
    if obs_rep[-1, i, -1, 3] > 10:
        obs_rep[-1, i, -1, 3] = 0

def plot_original(q=0, figsize=(8, 5), N=32):
    theory_function = [None, energy_theory, None, specHeat_theory]
    ylabel = ['$M$', '$E$', '$\chi $', '$C_V$']
    
    plt.figure(figsize=figsize)
    plt.plot(T_list_th, theory_function[q](T_list_th, N=N), color='magenta', label='Theory N=%d'%N)
    plt.plot(T_list, obs_or[:, 0, q], '--', color='blue', label='MC N=%d'%N)
    plt.plot(T_list, obs_or[:, -1, q], '*', color='red', label='SR N=%d'%N)
    
    plt.xlabel('$T$')
    plt.ylabel(ylabel[q])
    
    plt.show()
    
def plot_rep(q=0, figsize=(8, 5), N=32):
    logL_original = int(np.log2(N))
    N_list = 2 ** np.arange(logL_original+1, logL_original+len(obs_rep)+1)
    color_list = ['blue', 'red', 'green']
    
    ylabel = ['$M$', '$E$', '$\chi $', '$C_V$']
    
    T_ren = T_list
    plt.figure(figsize=figsize)
    plt.plot(T_list, obs_or[:, -1, q], '-*', color='magenta', label='MC N=%d'%N)
    for (i, N) in enumerate(N_list):
        T_ren = 2.0 / np.arccosh(np.exp(2.0 / T_ren))
        plt.plot(T_ren, obs_rep[i, :, -1, q], '-*', color=color_list[i],
                 label='SR N=%d'%N)
    
    plt.xlabel('$T$')
    plt.ylabel(ylabel[q])
    plt.legend()
        
    plt.show()
    
def plot_rep_th(q=0, figsize=(8, 5), N=32):
    logL_original = int(np.log2(N))
    N_list = 2 ** np.arange(logL_original+1, logL_original+len(obs_rep)+1)
    color_list = ['blue', 'red', 'green']
    
    theory_function = [None, energy_theory, None, specHeat_theory]
    ylabel = ['$M$', '$E$', '$\chi $', '$C_V$']
    
    T_ren = T_list
    plt.figure(figsize=figsize)
    for (i, N) in enumerate(N_list):
        T_ren = 2.0 / np.arccosh(np.exp(2.0 / T_ren))
        plt.plot(T_list_th, theory_function[q](T_list_th, N=N), color=color_list[i],
                 label='Theory N=%d'%N)
        plt.plot(T_ren, obs_rep[i, :, -1, q], '*', color=color_list[i],
                 label='SR N=%d'%N)
    
    plt.xlabel('$T$')
    plt.ylabel(ylabel[q])
    #plt.legend()
        
    plt.show()
    
