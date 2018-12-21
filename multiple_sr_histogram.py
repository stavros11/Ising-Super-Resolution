# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:49:37 2018

@author: Stavros
"""

import numpy as np
from plot.plot_directories import models_critical_dir
from networks.architectures import simple2D_pbc, duplicate_simple2D_pbc
class Ising():
    def __init__(self, state):
        (self.n_samples, self.Ly, self.Lx) = state.shape
        self.N_spins = self.Lx * self.Ly
        self.state = state
        
    def _calculate_magnetization(self):
        self.sample_mag = np.sum(self.state, axis=(1,2))
        
    def _calculate_energy(self, Jx=1, Jy=1):
        ## Returns total energy of the current state - Full calculation ##
        # Energy from x interactions
        Ex = np.sum(self.state[:,:,1:] * self.state[:,:,:self.Lx-1], axis=(1,2))
        # Energy from y interactions
        Ey = np.sum(self.state[:,1:,:] * self.state[:,:self.Ly-1,:], axis=(1,2))
        
        # Fix periodic boundary conditions
        Ex+= np.sum(self.state[:,:,0] * self.state[:,:,self.Lx-1], axis=1)
        Ey+= np.sum(self.state[:,0,:] * self.state[:,self.Ly-1,:], axis=1)
        
        self.sample_energy = - (Jx * Ex + Jy * Ey)
    
    def calculate_moments(self):
        self._calculate_magnetization()
        self.mag  = np.mean(np.abs(self.sample_mag))
        self.mag2 = np.mean(np.square(self.sample_mag))
        self.mag4 = np.mean(self.sample_mag**4)
        
        self._calculate_energy()
        self.energy  = np.mean(self.sample_energy)
        self.energy2 = np.mean(np.square(self.sample_energy))
        

## Load model ##
NAME = 'Simple2D16relu_L2_64_32_K333_PBC_MReg0.00EReg0.00B1000_Ver2Run7'
model0 = simple2D_pbc([1, 8, 8, 1], kernels=[3,3,3])
model0.load_weights('%s/%s.hdf5'%(models_critical_dir, NAME))
#
### Load data ##
n_samples = 32000
L_list = 2**np.arange(3, 7)
data_dir = 'D:\Ising-Super-Resolution-Data\ising-data-GPU'
data_dir+= '\L=%d\q=2\configs.npy'
data_mc = [np.load(data_dir%L).reshape(n_samples, L, L) for L in L_list]

mag_mc, energy_mc = [], []
for (L, x) in zip(L_list, data_mc):
    obj = Ising(2 * x - 1)
    obj.calculate_moments()
    mag_mc.append(np.abs(obj.sample_mag) / L**2)
    energy_mc.append(obj.sample_energy / L**2)
    

## Upsample starting from 8x8
mag_sr, energy_sr = [], []
print("Upsampling started")
states = [data_mc[0]]
cont = model0.predict(states[-1].reshape(states[-1].shape + (1,)))[:,:,:,0]
states.append((cont > np.random.random(cont.shape)).astype(np.int))
obj = Ising(2 * states[-1] - 1)
obj.calculate_moments()
mag_sr.append(np.abs(obj.sample_mag) / 16**2)
energy_sr.append(obj.sample_energy / 16**2)
for i in range(1, 3):
    model = duplicate_simple2D_pbc(model0, states[-1].shape + (1,), 
                                   hid_filters=[64, 32], kernels=[3, 3, 3])
    cont = model.predict(states[-1].reshape(states[-1].shape + (1,)))[:,:,:,0]
    states.append((cont > np.random.random(cont.shape)).astype(np.int))
    obj = Ising(2 * states[-1] - 1)
    obj.calculate_moments()
    mag_sr.append(np.abs(obj.sample_mag) / L_list[i+1]**2)
    energy_sr.append(obj.sample_energy / L_list[i+1]**2)
    print("%d done"%(i+1))
    
## Plots
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from seaborn import distplot, kdeplot, color_palette
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 40})

label_list = ['MC', 'SR']
alphas = [1.0, 0.8, 0.65]
text_font = 48

fig, axs = plt.subplots(figsize=(20,11), ncols=3, nrows=2)      

cp = color_palette("deep", 3)
for i in range(len(L_list[1:])):
    distplot(mag_mc[i+1], bins=15, kde=True, color=cp[0], label=label_list[0],
             norm_hist=True, hist_kws=dict(alpha=alphas[0]), ax=axs[0, i],
             kde_kws={"lw":4.0})
    distplot(mag_sr[i], bins=15, kde=True, color=cp[1], label=label_list[1],
             norm_hist=True, hist_kws=dict(alpha=alphas[1]), ax=axs[0, i],
             kde_kws={"lw":4.0})
        
    distplot(energy_mc[i+1], bins=15, kde=True, color=cp[0], label=label_list[0],
             norm_hist=True, hist_kws=dict(alpha=alphas[0]), ax=axs[1, i],
             kde_kws={"lw":4.0})
    distplot(energy_sr[i], bins=15, kde=True, color=cp[1], label=label_list[1],
             norm_hist=True, hist_kws=dict(alpha=alphas[1]), ax=axs[1, i],
             kde_kws={"lw":4.0})
        #kdeplot(obs_plot[i][l], rc={"lines.linewidth": 2.5}, color=color_list[l])
        
    axs[0, i].set_xlabel('$M$', fontsize=42)
    axs[1, i].set_xlabel('$E$', fontsize=42)
    axs[1, i].locator_params(axis='x', nbins=4)
    axs[1, i].locator_params(axis='x', nbins=4)
    
    axs[0, i].set_title("L=%d"%L_list[i+1], fontsize=42)

axs[0, 0].legend(fontsize=38, loc='upper left')

plt.tight_layout()
#plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.10)
plt.savefig('seaborn_multi.pdf', bbox_inches='tight')