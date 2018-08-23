# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:34:54 2018

@author: User
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

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

def tpf_theory(T, k, J=1, N=32):
    th = np.tanh(J / T)
    return (th**k + th**(N-k)) / (1 + th**N)

### !!! .NPY DESCRIPTION !!! ###
# obs = (32, 5, 12)
# ind1: temperatures
# ind2: [MC, RG, SR continuous, SR rounded, SR sampled]
# ind3: [Mag, En, Susc, specHeat, Mag2, Mag4, En2, tpf(L/2), 
#        tpf(L/4), S0, S1, S2]

# Load data (fix .npy directory here)
NAME = 'Simple1D32relu_L1_32_K53_PBC_MReg0.00EReg0.00B1000_TS5_UP4_VER4'
TPFD = 16

obs_or = np.load('%s/%s.npy'%(quantities_dir1D, NAME[:-13]))
obs_rep = np.load('%s/%s.npy'%(quantities_dir1D_rep, NAME))
tpf = np.load('%s/%s_TPF_vK16.npy'%(quantities_dir1D_rep, NAME[:-9]))

# Use rounding instead of sampling for the five lowest temperatures 
# to correct noise in susc and Cv
obs_or[:2, -1] = obs_or[:2, -2]
obs_rep[:, :3, -1] = obs_rep[:, :3, -2]

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
    
def plot_rep_th(q=0, figsize=(8, 5), N=32, xlims=[0, 4], ylims=[-1, -0.2]):
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
    plt.xlim(xlims)
    plt.ylim(ylims)
    #plt.legend()
        
    plt.show()
    
def plot_rep_tpf_th(figsize=(8, 5), N=32):
    logL_original = int(np.log2(N))
    N_list = 2 ** np.arange(logL_original+1, logL_original+len(obs_rep)+1)
    color_list = ['blue', 'red', 'green']
    
    T_ren = T_list
    plt.figure(figsize=figsize)
    for (i, N) in enumerate(N_list):
        T_ren = 2.0 / np.arccosh(np.exp(2.0 / T_ren))
        plt.plot(T_list_th, tpf_theory(T_list_th, k=int(N**0.8/5), N=N), color=color_list[i],
                 label='N=%d'%N)
        plt.plot(T_ren, tpf[i], '*', color=color_list[i])
    
    plt.xlabel('$T$')
    plt.ylabel('$C(T)$')
    #plt.legend()
        
    plt.show()
    
mc1d = np.load('C:/Users/Stavros/Documents/Scripts_and_programs/Ising_Data/ising1d-data-test-10000/ising-1d-N32-samples10000-test.npy')
def two_point_function(state, k):
    N = state.shape[1]
    copy = np.empty(state.shape)
    copy[:, :N-k] = state[:, k:]
    copy[:, N-k:] = state[:, :k]

    return (copy * state).mean()

logL_original = int(np.log2(32))
N_list = 2 ** np.arange(logL_original, logL_original+len(obs_rep)+1)

Tl = [T_list]
for i in range(len(N_list)-1):
    Tl.append(2.0 / np.arccosh(np.exp(2.0 / Tl[i])))

tpf_or = np.zeros([32, 17])
tpf_th = np.zeros([5, 32, 17])
k_list = np.arange(17)
for iT in range(32):
    for k in k_list:
        tpf_or[iT, k] = two_point_function(2 * mc1d[iT * 10000 : (iT+1) * 10000] - 1, k=k)
        for (i, N) in enumerate(N_list):
            tpf_th[i, iT, k] = tpf_theory(Tl[i][iT], N=N, k=k)
        
tpf_plot = np.zeros([len(N_list), 32, 17])
tpf_plot[0] = tpf_or

## Test what happens if you only use RG temperature equation ##
#T_ren = 2.0 / np.log(np.cosh(2.0 / T_list))
#for i in range(1, 5):
#    for iT in range(32):
#        ind = np.abs(T_ren - T_list[iT]).argmin()
#        print(ind)
#        tpf_plot[i, iT] = tpf_or[ind]
#    T_ren = 2.0 / np.log(np.cosh(2.0 / T_ren))
    
tpf_plot[1:] = tpf

obs_plot = np.zeros([len(N_list), 32, 7])
obs_plot[0] = obs_or[:, -1]
obs_plot[1:] = obs_rep[:,:,-1]

alpha_list = [1.0, 0.8, 0.6, 0.4, 0.2]
marker_list = ['s', '^', 'o', 'd', 'v']
color_list = ['blue', 'red', 'green', 'magenta', 'black']

#from mpl_toolkits.axes_grid.inset_locator import inset_axes

plt.figure(figsize=(30, 8))
gs = gridspec.GridSpec(1, 2)
matplotlib.rcParams.update({'font.size': 32})
label_size = 40
text_size = 52


plt.subplot(gs[0])
iT = 10
for (i, N) in enumerate(N_list):
    plt.plot(k_list, tpf_th[i, iT], color='blue', alpha=alpha_list[i], linewidth=4.5, label='$N=%d$'%N)
    
    plt.plot(k_list, tpf_plot[i, iT], linestyle='', color='red', alpha=alpha_list[i], linewidth=3.0,
             marker=marker_list[i], markersize=16)
    
plt.xlabel('$j$', fontsize=label_size)
plt.ylabel('$G_N(j)$', fontsize=label_size)
plt.legend(loc='lower left', fontsize=32)

plt.text(16, 0.98, 'a', horizontalalignment='center', verticalalignment='center', 
                 fontweight='bold', fontsize=text_size)


ax = plt.subplot(gs[1])
iT = 20
for (i, N) in enumerate(N_list):
    plt.plot(k_list, tpf_th[i, iT], color='blue', alpha=alpha_list[i], linewidth=4.5, label='$N=%d$'%N)
    
    plt.plot(k_list, tpf_plot[i, iT], linestyle='', color='red', alpha=alpha_list[i], linewidth=3.0,
             marker=marker_list[i], markersize=16)
    
plt.xlabel('$j$', fontsize=label_size)
plt.ylabel('$G_N(j)$', fontsize=label_size)

plt.text(16, 0.98, 'b', horizontalalignment='center', verticalalignment='center', 
                 fontweight='bold', fontsize=text_size)

#inset_axes = inset_axes(ax, width="40%", height="30%", loc='upper right')

plt.savefig('extrapolation1D.pdf', bbox_inches='tight')

#plt.figure(figsize=(18, 10))
#for i in range(1, len(N_list)):
#    N = N_list[i]
#    plt.plot(T_list_th, energy_theory(T_list_th, N=N), color='blue', alpha=alpha_list[i], linewidth=4.5)
#    
#    plt.plot(Tl[i], obs_plot[i, :, 1], linestyle='', color='red', alpha=alpha_list[i], linewidth=3.0,
#             marker=marker_list[i], markersize=16, label='$N=%d$'%N)
#    
#plt.xlabel('$T$', fontsize=label_size)
#plt.ylabel('$E$', fontsize=label_size)
#plt.legend(loc='upper left')
#
#plt.xlim([0.1, 1])
#plt.ylim([-1, -0.75])
#
#plt.savefig('energy1D.pdf', bbox_inches='tight')    
