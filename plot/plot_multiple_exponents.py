# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:35:29 2018

@author: Stavros
"""

import numpy as np
from plot_directories import multiple_exponents_dir

NAME = 'Simple2D16relu_L2_64_32_K333_PBC_C2UP3'

ev = np.load('%s/%s.npy'%(multiple_exponents_dir, NAME))