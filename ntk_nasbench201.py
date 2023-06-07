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

def main(xargs):
    PID = os.getpid()
    assert torch.cuda.is_available(), 'CUDA is not available.'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_seed(xargs.rand_seed)

    if xargs.timestamp == 'none':
        xargs.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)

    ##### config & logging #####
    config = edict()
    config.class_num = class_num
    config.xshape = xshape
    config.batch_size = xargs.batch_size
    xargs.save_dir = "./output/ntk/%s"%(xargs.dataset)+ "/repeat%d"%(xargs.repeat) + "/{:}/seed{:}".format(xargs.timestamp, xargs.rand_seed)
    config.save_dir = xargs.save_dir
    logger = prepare_logger(xargs)
    ###############
    api = API(xargs.arch_nas_dataset)
    num = len(api)
    #for i, arch_str in enumerate(api):
    #  print ('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))

    #if xargs.dataset != 'imagenet-1k':
    #    search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/', xargs.batch_size, xargs.workers)
    #else:
    #    train_loader = torch.utils.data.DataLoader(train_data, batch_size=xargs.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    with open(xargs.save_dir+'/results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['index','ntk','acc'])
    train_loader  = torch.utils.data.DataLoader(train_data, batch_size=xargs.batch_size, shuffle=True, num_workers=xargs.workers, pin_memory=True)
    ntk_delta = []
    ntk_all = []
    repeat = xargs.repeat
    for index in range(100):
        config = api.get_net_config(index, 'cifar100') # obtain the network configuration for the 123-th architecture on the CIFAR-10 dataset
        network = get_cell_based_tiny_net(config) # create the network from configurration
        for _ in range(repeat):
            ntk = get_ntk_n(train_loader, [network, network], recalbn=0, train_mode=True, num_batch=1)
        ntk=1/np.mean(ntk)
        info= api.query_meta_info_by_index(index)
        res_metrics = info.get_metrics('cifar100', 'x-test')
        #res_metrics = info.get_metrics('ImageNet16-120', 'x-test')
        tmp =[]
        tmp.append(index)
        tmp.append(ntk)
        tmp.append(res_metrics["accuracy"])
        print(tmp)
        with open(xargs.save_dir+'/results.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(tmp) 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("TENAS")
    parser.add_argument('--data_path', type=str, default='./ssd1/cifar-100-python',help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')
    parser.add_argument('--search_space_name', type=str, default='nas-bench-201',  help='space of operator candidates: nas-bench-201 or darts.')
    parser.add_argument('--max_nodes', type=int, help='The maximum number of nodes.')
    parser.add_argument('--track_running_stats', type=int, choices=[0, 1], help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for ntk')
    parser.add_argument('--save_dir', type=str, help='Folder to save checkpoints and log.')
    parser.add_argument('--arch_nas_dataset', type=str, default='/home/haochx5/TENAS/ssd1/NAS-Bench-201-v1_0-e61699.pth' ,help='The path to load the nas-bench-201 architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--rand_seed', type=int, help='manual seed')
    parser.add_argument('--precision', type=int, default=3, help='precision for % of changes of cond(NTK) and #Regions')
    parser.add_argument('--prune_number', type=int, default=1, help='number of operator to prune on each edge per round')
    parser.add_argument('--repeat', type=int, default=3, help='repeat calculation of NTK and Regions')
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')
    parser.add_argument('--init', default='kaiming_uniform', help='use gaussian init')
    parser.add_argument('--super_type', type=str, default='basic',  help='type of supernet: basic or nasnet-super')
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    main(args)