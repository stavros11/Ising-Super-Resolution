# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:28:41 2018

@author: Stavros
"""

import numpy as np

def curve(x, a, b):
    return a / np.log(np.cosh(b / x))

def inv_curve(x, a, b):
    return b / np.arccosh(np.exp(a / x))