# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 20:27:23 2018

@author: Stavros
"""

import numpy as np

#####################################################
############ Functions that give metrics ############
######################################################
    
def get_metrics_one(hist, reg=False, val=False):
    val_str = ''
    if val:
        val_str = 'val_'
    # Takes metric object and returns the four metric arrays
    metrics = [hist.history[val_str+'loss']]
    metrics.append(hist.history[val_str+'cont_loss'])
    metrics.append(hist.history[val_str+'round_loss'])
    metrics.append(hist.history[val_str+'acc'])
    if reg:
        metrics.append(hist.history[val_str+'regularization'])

    
    return np.array(metrics)

def get_metrics(hist, reg=False):
    m1 = get_metrics_one(hist, reg=reg)
    m2 = get_metrics_one(hist, reg=reg, val=True)
    
    return np.array([m1, m2])