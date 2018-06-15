# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 21:09:15 2018

@author: Stavros
"""

from os import path, mkdir
from architectures import get_name_and_model
from calculators import round_loss, cont_loss
from metrics import get_metrics
from keras.callbacks import History, EarlyStopping

def create_directory(d):
    ## Create directory if it doesn't exist ##
    if not path.exists(d):
        mkdir(d)

class TrainerCritical():
    def __init__(self, args):
        self.args = args

    def create_loss(self):
        self.metrics_list = [round_loss, cont_loss, 'accuracy']
        if self.args.magR == 0 and self.args.enR == 0:
            self.reg_flag = False
            from calculators import create_loss
            def loss(self, y_true, y_pred):
                return create_loss(y_true, y_pred, ce=self.args.CE)
        else:
            self.reg_flag = True
            from calculators import create_loss_reg, regularization
            self.metrics_list.append(regularization)
            def loss(self, y_true, y_pred):
                return create_loss_reg(y_true, y_pred, ce=self.args.CE, 
                                       mag_reg=self.args.magR, en_reg=self.args.enR)
                
    def create_callbacks(self):
        self.callbacks = [History()]
        if self.args.ES:
            self.callbacks.append(EarlyStopping(monitor='loss', 
                                                min_delta=self.args.ESdelta, 
                                                patience=self.args.ESpat))
            self.args.EP = 1000

    def train(self, data):
        self.create_loss()
        self.create_callbacks()
        
        self.name, self.model = get_name_and_model(data.train_in.shape, 
                                                   hid_act=self.args.ACT,
                                                   hid_filters=self.args.HF, 
                                                   kernels=self.args.K,
                                                   pbc=self.args.PBC)
        
        self.model.compile(optimizer=self.args.OPT, loss=self.loss, 
                           metrics=self.metrics_list)
        
        hist = self.model.fit(x=data.train_in, y=data.train_out,
                              batch_size=self.args.BS, epochs=self.args.EP, 
                              verbose=0, callbacks=self.callbacks, 
                              validation_data=(data.test_in, data.test_out))
        
        self.metrics = get_metrics(hist, reg=self.reg_flag)
        