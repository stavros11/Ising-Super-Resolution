# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:54:48 2018

@author: Stavros
"""

from os import listdir
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-CR', type=bool, default=False, help='critical data')
args = parser.parse_args()

if args.CR:
    from data.directories import models_critical_save_dir as models_dir
else:
    from data.directories import models_save_dir as models_dir

print(args.CR)
print(listdir(models_dir))