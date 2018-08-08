# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 22:51:24 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
from networks.ising import get_observables_with_corr_and_tpf
from data.decimations import block_rg_WD
from scipy.optimize import curve_fit
from renormalization.curves import curve, inv_curve

def read_file(L=16, q=2, n_samples=10000, train=False):
        dr = 'C:/Users/Stavros/Documents/Scripts_and_programs/Ising_Data'
        dr += '/ising-data-%s-%d'%(['train', 'test'][int(train)], n_samples)
        dr += '/L=%d/q=%d/configs.npy'%(L, q)
        
        data = np.load(dr)
        return data.reshape(len(data), L, L)
    
def temp_partition(data, iT, n_samples=10000):
    return data[iT * n_samples : (iT+1) * n_samples]

def calculate_observables(mc_large, mc_small, T_list=np.linspace(0.01, 4.538, 32)):
    # [MC Large, RG, MC small]
    n_temps = len(T_list)
    obs = np.zeros([3, n_temps, 12])
    for (iT, T) in enumerate(T_list):
        obs[0, iT] = get_observables_with_corr_and_tpf(temp_partition(mc_large, iT), T)
        obs[1, iT] = get_observables_with_corr_and_tpf(block_rg_WD(temp_partition(mc_large, iT)), T)
        obs[2, iT] = get_observables_with_corr_and_tpf(temp_partition(mc_small, iT), T)
            
        print('Temperature %d / %d calculated!'%(iT+1, n_temps))
        
    return obs

class MCvsRG():
    def __init__(self, obs, L=16, T_list=np.linspace(0.01, 4.538, 32)):
        self.Tc = 2 / np.log(1 + np.sqrt(2))
        self.T_list = T_list
        self.L = L
       
        self.obs = obs
        
        ## For plots
        self.color_list = ['blue', 'red', 'green']
        self.label_list = ['MC %dx%d'%(self.L, self.L), 'RG %dx%d'%(self.L//2, self.L//2), 
                           'MC %dx%d'%(self.L//2, self.L//2)]
                    
    def plot_quantity(self, q=0, figsize=(6, 4), linestyle='-*'):
        plt.figure(figsize=figsize)
        for i in range(3):
            plt.plot(self.T_list, self.obs[i, :, q], linestyle, color=self.color_list[i], 
                     label=self.label_list[i])
        plt.axvline(x=self.Tc, linestyle='--', color='black')
        plt.legend()
        
        plt.show()
        
    def plot_four(self, figsize=(10, 6), linestyle='-*'):
        plt.figure(figsize=figsize)
        for q in range(4):
            plt.subplot(221+q)
            for i in range(3):
                plt.plot(self.T_list, self.obs[i, :, q], linestyle, color=self.color_list[i], 
                         label=self.label_list[i])
        plt.axvline(x=self.Tc, linestyle='--', color='black')
        plt.legend()
        
        plt.show()
        
    def transformation_points(self, mc, rg):
        # Neglect RG points beyond the known MC values
        i_valid = np.where((rg >= mc.min()) & (rg <= mc.max()))[0]
        # Neglect initial indices where RG is constant
        i_const = np.where(rg[1:] == rg[:-1])[0]
        i_valid = i_valid[np.where(i_valid > i_const[-1]+1)[0]]
        
        points = []
        for i in i_valid:
            i1, i2 = np.abs(mc - rg[i]).argsort()[:2]
            points.append(self.T_list[i1] + (self.T_list[i2] - self.T_list[i1]) * (rg[i] - mc[i1]) / (mc[i2] - mc[i1]))
            
        return np.array([self.T_list[i_valid], points])
    
    def fit_transformation(self):
        self.points = [self.transformation_points(self.obs[2, :, i], self.obs[1, :, i]) for i in range(2)]
        self.opt_params = [curve_fit(curve, p[0], p[1])[0] for p in self.points]
        
    def fitted_curve(self, x, q=0):
        return curve(x, a=self.opt_params[q][0], b=self.opt_params[q][1])
    
    def inverse_fitted_curve(self, x, q=0):
        return inv_curve(x, a=self.opt_params[q][0], b=self.opt_params[q][1])
        
    def plot_fit(self, q=0, tmin=0, tmax=5, figsize=(6, 4)):
        plt.figure(figsize=figsize)
        plt.scatter(self.points[q][0], self.points[q][1])
        
        tt = np.linspace(0, 5, 1000)
        ff = self.fitted_curve(tt, q=q)
        plt.plot(tt, ff)
        plt.show()
        
    def plot_quantity_fixed(self, q=0, figsize=(6, 4), linestyle='-*'):        
        plt.figure(figsize=figsize)
        for i in range(3):
            if i == 1:
                plt.plot(self.fitted_curve(self.T_list, q=q), self.obs[i, :, q], linestyle, color=self.color_list[i], 
                         label=self.label_list[i])
            else:
                plt.plot(self.T_list, self.obs[i, :, q], linestyle, color=self.color_list[i], 
                         label=self.label_list[i])
        plt.axvline(x=self.Tc, linestyle='--', color='black')
        plt.legend()
        
        plt.show()
        
L = 16
mc_large = read_file(L=L)
mc_small = read_file(L=L//2)

## Calculate observables
obs = calculate_observables(mc_large=mc_large, mc_small=mc_small)
np.save('MCvsRG_Observables_L%d.npy'%L, obs)

x = MCvsRG(obs=obs, L=L)
x.fit_transformation()

## Save transformations
np.save('Magnetization_Transformation_Points_L%d.npy'%L, x.points[0].T)
np.save('Energy_Transformation_Points_L%d.npy'%L, x.points[1].T)
np.save('Magnetization_Transformation_Params_L%d.npy'%L, x.opt_params[0])
np.save('Energy_Transformation_Params_L%d.npy'%L, x.opt_params[1])

