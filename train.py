# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:18:23 2018

@author: Stavros
"""

from data.directories import models_critical_save_dir, metrics_critical_save_dir
from data.loaders import TrainingData
from networks.train import TrainerCritical
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-CR', type=bool, default=False, help='critical data')
parser.add_argument('-RGWD', type=bool, default=False, help='well defined RG')
parser.add_argument('-L', type=int, default=16, help='output size')
parser.add_argument('-nTR', type=int, default=40000, help='train samples')
parser.add_argument('-nTE', type=int, default=100000, help='test samples')
parser.add_argument('-TRS', type=int, default=40000, help='train size')
parser.add_argument('-VALS', type=int, default=5000, help='validation size')

parser.add_argument('-PBC', type=bool, default=False, help='use PBC padding')
parser.add_argument('-ACT', type=str, default='relu', help='hidden activation')
parser.add_argument('-HF', type=list, default=[64,32], help='hidden filters list')
parser.add_argument('-K', type=list, default=[6,1,3], help='kernel size list')

parser.add_argument('-OPT', type=str, default='opt', help='optimizer')
parser.add_argument('-CE', type=bool, default=False, help='use cross-entropy')
parser.add_argument('-magR', type=float, default=0.0, help='magnetization regularization')
parser.add_argument('-enR', type=float, default=0.0, help='energy regularization')

parser.add_argument('-EP', type=int, default=100, help='epochs')
parser.add_argument('-ES', type=bool, default=False, help='early stopping')
parser.add_argument('-ESpat', type=int, default=50, help='early stopping patience')
parser.add_argument('-ESdelta', type=float, default=0.0001, help='early stopping delta')

args = parser.parse_args()
args.models_dir = models_critical_save_dir
args.metrics_dir= metrics_critical_save_dir

trainer = TrainerCritical(args)
trainer.train(TrainingData(args))
