import os, sys, time, argparse
import math
import random
import numpy as np
import torch
from easydict import EasyDict as edict
from torch import nn
from pathlib import Path
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from nas_201_api import NASBench201API as API
from datasets import get_datasets, get_nas_search_loaders
from models import get_cell_based_tiny_net
from procedures import prepare_seed, prepare_logger
from procedures import Linear_Region_Collector, get_ntk_n
import time, argparse, csv
import scipy.stats as stats

File='/home/haochx5/TENAS/output/ntk/cifar100/repeat3/May-23-20_05-10-1684847447/seed25048/results.csv'
ntk=[]
acc=[]
ntk1=[]
acc1=[]
with open(File,newline='') as csvfile:
    spamreader =csv.DictReader(csvfile)
    for row in spamreader:
        tmp1=row['ntk']
        tmp2=row['acc']
        ntk.append(1/float(tmp1))
        acc.append(float(tmp2))
        ntk1.append(tmp1)
        acc1.append(tmp2)
tau, p =stats.kendalltau(ntk,acc)
print(tau)
t,p=stats.kendalltau(ntk1,acc1)
print(t)

