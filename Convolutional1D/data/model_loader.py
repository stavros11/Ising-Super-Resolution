# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 16:38:28 2018

@author: Stavros
"""

from os import listdir
from directories import models_save_dir as model_dir
from keras.losses import mean_squared_error
from keras.models import load_model

############################
### CLASS TO LOAD MODELS ###
############################

class ModelLoader():
    def __init__(self, list_ind):        
        self.model_dir = model_dir
        self.name = listdir(model_dir)[list_ind]
        self.regularization = self.reg_from_name(self.name)
        
        metrics_list = ['round_loss', 'cont_loss']
        
        if self.regularization:
            metrics_list.append('regularization')
            
        self.custom_objects = {'loss': mean_squared_error}
        for x in metrics_list:
            self.custom_objects[x] = mean_squared_error
        
        if self.name[-3:] == 'OLD':
            self.priorT_symbol = 'T'
        else:
            self.priorT_symbol = 'Net'
            
    def update_temperature(self, T):
        self.graph = load_model('%s/%s/%s%.4f.h5'%(self.model_dir, self.name, self.priorT_symbol, T), 
                                custom_objects=self.custom_objects)
            
    @staticmethod
    def reg_from_name(name):
        i = 4
        while name[i:i+4] != 'MReg':
            i += 1
        i += 4
        mag_reg = float(name[i:i+4])
        i += 8
        en_reg = float(name[i:i+4])
        
        if mag_reg == 0.0 and en_reg == 0.0:
            return False
        else:
            return True