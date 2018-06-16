# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 12:00:24 2018

@author: Stavros
"""

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def set_GPU_memory(fraction=0.3):    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    set_session(tf.Session(config=config))